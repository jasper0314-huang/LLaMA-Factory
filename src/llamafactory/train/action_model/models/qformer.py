import torch
import torch.nn as nn
from timm.layers import Mlp


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
        with_film = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.queries = nn.Parameter(0.02 * torch.randn(1, num_queries, embed_dim))
        self.with_film = with_film

        self.blocks = nn.ModuleList([
            XBlock(
                embed_dim,
                cross_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout_rate,
                norm_layer=norm_layer,
                with_film = self.with_film,
            )
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, context, language_embedding=None):
        bs = context.shape[0]
        x = self.queries.expand(bs, -1, -1)
        for blk in self.blocks:
            x = blk(x, context, language_embedding=language_embedding)
        x = self.norm(x)
        return x


class XBlock(nn.Module):

    def __init__(
        self,
        dim,
        kv_dim=-1,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_film = False
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
        )
        if kv_dim < 0:
            kv_dim = dim
        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias,
            add_bias_kv=qkv_bias,
            batch_first=True,
            kdim=kv_dim,
            vdim=kv_dim,
        )
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=dropout)

        self.with_film = with_film
        if self.with_film:
            # print(f'kv_dim: {kv_dim}', flush=True) # 768
            self.film_layer = FilmConditioning(in_dim = kv_dim, num_channels = dim)

    def forward(self, x, context, language_embedding=None):
        # self-attention
        norm_x = self.norm1_1(x)
        x = x + self.self_attn(norm_x, norm_x, norm_x, need_weights=False)[0]
        # cross-attention
        norm_x = self.norm1_2(x)
        x = x + self.cross_attn(norm_x, context, context, need_weights=False)[0]
        # mlp
        x = x + self.mlp(self.norm2(x))

        if self.with_film:
            assert language_embedding is not None
            x = self.film_layer(x, language_embedding)

        return x


class FilmConditioning(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_channels: int,
        ):
        super().__init__() 
        self.proj_mul = nn.Linear(in_dim, num_channels)
        self.proj_add = nn.Linear(in_dim, num_channels)
        nn.init.zeros_(self.proj_mul.weight)
        nn.init.zeros_(self.proj_mul.bias)
        nn.init.zeros_(self.proj_add.weight)
        nn.init.zeros_(self.proj_add.bias)

    def forward(self, image_features, language_embeddings):
        """
        image_features: [B, 32, D]
        language_embeddings: [B, L, D]
        """
        language_embeddings = language_embeddings.mean(dim=1)

        mul_feat  = self.proj_mul(language_embeddings).unsqueeze(1)   # [bs, 1, 768]
        add_feat = self.proj_add(language_embeddings).unsqueeze(1)    # [bs, 1, 768]

        image_features = (1 + mul_feat) * image_features + add_feat
        return image_features
