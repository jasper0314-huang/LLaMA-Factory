import einops
import numpy as np
import math
import torch
import torch.nn as nn
from timm.layers import DropPath, Mlp
from torch.utils.checkpoint import checkpoint

import einops
import numpy as np
from einops import rearrange


class FilmConditioning(nn.Module):
    """Layer that adds FiLM conditioning.

    This is intended to be applied after a convolutional layer. It will learn a
    multiplicative and an additive factor to be applied to each channel of the
    convolution's output.

    Conv layer can be rank 2 or 4.

    For further details, see: https://arxiv.org/abs/1709.07871
    """
    
    def __init__(self,
                 in_dim: int,
                 num_channels: int):
        """Constructs a FiLM conditioning layer.

        Args:
            num_channels: Number of filter channels to expect in the input.
        """
        super(FilmConditioning, self).__init__() 
        # Note that we initialize with zeros because empirically we have found
        # this works better than initializing with glorot.
        self._projection_add = nn.Linear(in_dim, num_channels)
        self._projection_mult = nn.Linear(in_dim, num_channels)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, conv_filters, conditioning):
        assert len(conditioning.shape) == 2
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_mult = self._projection_mult(conditioning)

        if len(conv_filters.shape) == 4:
            # [B, D] -> [B, D, 1, 1]
            projected_cond_add = projected_cond_add.unsqueeze(-1).unsqueeze(-1)
            projected_cond_mult = projected_cond_mult.unsqueeze(-1).unsqueeze(-1)
        elif len(conv_filters.shape) == 3:
            # [B, D] -> [B, 1, D]
            projected_cond_add = projected_cond_add.unsqueeze(1)
            projected_cond_mult = projected_cond_mult.unsqueeze(1)
        else:
            assert len(conv_filters.shape) == 2
        
        # Original FiLM paper argues that 1 + gamma centers the initialization at
        # identity transform.
        result = (1 + projected_cond_mult) * conv_filters + projected_cond_add
        return result


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        qk_normalization=False,
        qk_normalization_head_merged=False,
        causal=False,
        max_position_embeddings=-1,
    ):
        super().__init__()
        self.attn_drop_rate = attn_drop
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_normalization = qk_normalization
        self.qk_normalization_head_merged = qk_normalization_head_merged
        qk_norm_dim = dim if qk_normalization_head_merged else head_dim
        self.q_norm = norm_layer(qk_norm_dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(qk_norm_dim) if qk_normalization else nn.Identity()
        self.causal = causal
        self.rotary_emb = None

    def forward(self, x):
        B, L, C = x.shape

        def _qk_norm(q, k, head_first=True, dtype=torch.float):
            # qk normalization proposed in ViT-22B paper
            if not head_first:
                q = rearrange(q, "B L H D -> B H L D")
                k = rearrange(k, "B L H D -> B H L D")

            if self.qk_normalization:
                if self.qk_normalization_head_merged:
                    # B, H, N, D
                    B_, H_, N_, D_ = q.shape
                    q = self.q_norm(rearrange(q, "B H L D -> B L (H D)"))
                    q = rearrange(q, "B L (H D) -> B H L D", H=H_)
                    k = self.k_norm(rearrange(k, "B H L D -> B L (H D)"))
                    k = rearrange(k, "B L (H D) -> B H L D", H=H_)
                else:
                    q = self.q_norm(q)
                    k = self.k_norm(k)

            if not head_first:
                q = rearrange(q, "B H L D -> B L H D")
                k = rearrange(k, "B H L D -> B L H D")
            return q.to(dtype=dtype), k.to(dtype=dtype)

        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k, v = qkv.unbind(0)  # B H L D  # make torchscript happy (cannot use tensor as tuple)
        q, k = _qk_norm(q, k, dtype=v.dtype)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
        if self.causal:
            attn_mask = torch.full_like(attn, fill_value=float("-inf")).triu_(diagonal=1).to(dtype=attn.dtype)
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        kv_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        qk_normalization=False,
        qk_normalization_head_merged=False,
    ):
        super().__init__()

        self.attn_drop_rate = attn_drop
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_normalization = qk_normalization
        self.qk_normalization_head_merged = qk_normalization_head_merged
        qk_norm_dim = dim if qk_normalization_head_merged else head_dim
        self.q_norm = norm_layer(qk_norm_dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(qk_norm_dim) if qk_normalization else nn.Identity()

    def forward(self, x, context):
        B, L, C = x.shape

        def _qk_norm(q, k, head_first=True, dtype=torch.float):
            # qk normalization proposed in ViT-22B paper
            if not head_first:
                q = rearrange(q, "B L H D -> B H L D")
                k = rearrange(k, "B L H D -> B H L D")

            if self.qk_normalization:
                if self.qk_normalization_head_merged:
                    # B, H, N, D
                    B_, H_, N_, D_ = q.shape
                    q = self.q_norm(rearrange(q, "B H L D -> B L (H D)"))
                    q = rearrange(q, "B L (H D) -> B H L D", H=H_)
                    k = self.k_norm(rearrange(k, "B H L D -> B L (H D)"))
                    k = rearrange(k, "B L (H D) -> B H L D", H=H_)
                else:
                    q = self.q_norm(q)
                    k = self.k_norm(k)

            if not head_first:
                q = rearrange(q, "B H L D -> B L H D")
                k = rearrange(k, "B H L D -> B L H D")
            return q.to(dtype=dtype), k.to(dtype=dtype)

        q = self.q(x)
        kv = self.kv(context)
        # q = einops.rearrange(q, "B L (H D) -> B L H D", H=self.num_heads)
        q = einops.rearrange(q, "B L (H D) -> B H L D", H=self.num_heads)
        kv = einops.rearrange(kv, "B L (K H D) -> K B H L D", K=2, H=self.num_heads)
        k, v = kv.unbind(0)  # B H L D  # make torchscript happy (cannot use tensor as tuple)
        q, k = _qk_norm(q, k, dtype=v.dtype)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        qk_normalization=False,
        qk_normalization_head_merged=False,
        causal=False,
        max_position_embeddings=-1,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
            qk_normalization_head_merged=qk_normalization_head_merged,
            causal=causal,
            max_position_embeddings=max_position_embeddings,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, return_attention=False):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, return_attention)
        else:
            return self._forward(x, return_attention)

    def _forward(self, x, return_attention=False):
        if return_attention:
            _, attn = self.attn(self.norm1(x))
            return attn
        else:
            y, _ = self.attn(self.norm1(x))
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class XBlock(nn.Module):

    def __init__(
        self,
        dim,
        kv_dim=-1,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        qk_normalization=False,
        qk_normalization_head_merged=False,
        causal=False,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        with_film = False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
            qk_normalization_head_merged=qk_normalization_head_merged,
            causal=causal,
        )
        if kv_dim < 0:
            kv_dim = dim
        self.cross_attn = CrossAttention(
            dim,
            kv_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
            qk_normalization_head_merged=qk_normalization_head_merged,
        )
        self.norm1_1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_checkpoint = use_checkpoint

        self.with_film = with_film
        if self.with_film:
            # print(f'kv_dim: {kv_dim}', flush=True) # 768
            self.film_layer = FilmConditioning(in_dim = kv_dim, num_channels = dim)

    def forward(self, x, context, language_embedding = None):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, context, language_embedding)
        else:
            return self._forward(x, context, language_embedding)

    def _forward(self, x, context, language_embedding = None):
        y, _ = self.self_attn(self.norm1(x))
        x = x + self.drop_path(y)

        y, _ = self.cross_attn(self.norm1_1(x), context)
        x = x + self.drop_path(y)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.with_film == True:
            assert language_embedding is not None
            x = self.film_layer(x, language_embedding)

        return x


class QFormer(nn.Module):
    def __init__(
        self,
        num_queries=32,
        embed_dim=768,
        cross_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        dropout_rate=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        causal=False,
        qk_normalization=False,
        qk_normalization_head_merged=False,
        ctx2query=False,
        with_cls_token=False,
        query_type="default",
        dynamic_query="",
        num_queries_min=1,
        anneal_step=-1,
        with_film = False,
    ):
        super().__init__()
        self.dynamic_query = dynamic_query
        self.num_queries_min = num_queries_min
        self.anneal_step = anneal_step
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.with_cls_token = with_cls_token
        self.ctx2query = ctx2query

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        if query_type == "embed":
            self.queries = nn.Embedding(num_queries, embed_dim)
        else:
            self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        self.with_film = with_film

        self.blocks = nn.ModuleList(
            [
                XBlock(
                    embed_dim,
                    cross_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=None,
                    drop=dropout_rate,
                    attn_drop=dropout_rate,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    causal=causal,
                    qk_normalization=qk_normalization,
                    qk_normalization_head_merged=qk_normalization_head_merged,
                    with_film = self.with_film
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

        self.register_buffer("num_iter", torch.zeros(1))

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(self.queries, nn.Embedding):
            torch.nn.init.normal_(self.queries.weight, std=0.02)
        else:
            torch.nn.init.normal_(self.queries, std=0.02)

        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def sample_queries(self, num_queries=None):
        if isinstance(self.queries, nn.Embedding):
            queries = self.queries.weight[None, :]
        else:
            queries = self.queries

        if not self.training:
            if num_queries is not None:
                queries = queries[:, :num_queries]
            return queries

        # from IPython import embed
        # embed(header='sample q 1.')
        if self.dynamic_query == "uniform":
            num_queries_min = self.num_queries_min
            if self.anneal_step > 0 and self.num_iter < self.anneal_step:
                num_queries_min = int(self.num_queries - self.num_iter / self.anneal_step * (self.num_queries - self.num_queries_min))
            num_queries = torch.randint(num_queries_min, self.num_queries + 1, (1,))
            queries = queries[:, :num_queries]
        elif self.dynamic_query == "square":
            sqrt_q = int(self.num_queries**0.5)
            sqrt_q_min = int(self.num_queries_min**0.5)
            num_queries_list = [q**2 for q in range(sqrt_q_min, sqrt_q + 1)]
            num_queries = np.random.choice(num_queries_list)
            queries = queries[:, :num_queries]

        return queries

    def forward(self, context, language_embedding = None, num_queries=None, **kwargs):
        B, L, C = context.shape
        queries = self.sample_queries(num_queries=num_queries)
        x = queries.expand(B, -1, -1)
        # x = x + self.pos_embed

        if self.ctx2query:
            if self.with_cls_token:
                assert self.num_queries == L - 1
                cls_token = context[:, :1, :]
                x = x + cls_token + context[:, 1:, :]
            else:
                assert self.num_queries == L
                x = x + context

        for blk in self.blocks:
            x = blk(x, context, language_embedding = language_embedding)

        x = self.norm(x)

        self.num_iter += 1

        return x


if __name__ == "__main__":
    qformer = QFormer(
        num_queries=32,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        dropout_rate=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        with_film = True,
    )
    x = torch.randn((2, 257, 768), device="cuda")
    language_embedding = torch.randn((2, 768), device = "cuda")
    qformer = qformer.to(device="cuda")
    y = qformer(x, language_embedding)
    print(y.shape)
