# Modified from CogACT (https://github.com/microsoft/CogACT) repo

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


###############################################################
#               Embedding Layers for Timesteps                #
###############################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with self-attention conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=7,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        text_emb_dim=4096,
        img_emb_dim=4096,
        future_action_window_size=1,
        past_action_window_size=0,
        num_image_tokens=32,
    ):
        super().__init__()

        assert past_action_window_size == 0, "Error: action_history is not used now"

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.pred_action_window_size = future_action_window_size + 1

        self.time_embedder = TimestepEmbedder(hidden_size)

        self.action_proj = nn.Linear(in_channels, hidden_size)
        self.image_proj = nn.Linear(img_emb_dim, hidden_size)
        self.text_proj = nn.Linear(text_emb_dim, hidden_size)

        # Learnable positional embeddings
        # num of predict actions, time embedding, image tokens, and text tokens
        scale = hidden_size ** -0.5
        num_tokens = self.pred_action_window_size + 1 + num_image_tokens + 77
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_tokens, hidden_size))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize token_embed like nn.Linear
        nn.init.normal_(self.action_proj.weight, std=0.02)
        nn.init.constant_(self.action_proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.image_proj.weight, std=0.02)
        nn.init.constant_(self.image_proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.text_proj.weight, std=0.02)
        nn.init.constant_(self.text_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, text_features, image_features, addit_cond_embeddings=None):
        """
        Forward pass of DiT.
        x: (N, T, D) tensor of predicting action inputs
        t: (N,) tensor of diffusion timesteps
        text_features: (N, L_T, D) tensor of text features
        image_features: (N, L_I, D) tensor of image features
        addit_cond_embeddings: (N, L, D) tensor of additional conditioning embeddings
        """
        assert x.shape[1] == self.pred_action_window_size
        x = self.action_proj(x)                                 # (N, T, D)
        t = self.time_embedder(t).unsqueeze(1)                  # (N, 1, D)
        z_text = self.text_proj(text_features)                  # (N, L_T, D)
        z_image = self.image_proj(image_features)               # (N, L_I, D)
        x = torch.cat((x, t, z_image, z_text), dim=1)           # (N, L', D)
        x = x + self.positional_embedding                       # (N, L', D)
        if addit_cond_embeddings is not None:
            x = torch.cat((x, addit_cond_embeddings), dim=1)    # (N, L', D)
        for block in self.blocks:
            x = block(x)                                        # (N, L', D)
        x = self.final_layer(x)                                 # (N, L', out_channels)
        return x[:, :self.pred_action_window_size, :self.in_channels]
