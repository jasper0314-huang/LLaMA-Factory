# Modified from CogACT (https://github.com/microsoft/CogACT) repo

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPTextModel
from typing import Dict, List
from timm.models import create_model

from ....model.loader import register_custom_model

from . import create_diffusion
from . import gaussian_diffusion as gd
from .models import DiT
from .qformer import QFormer

# Create model sizes of ActionModels
def DiT_S():
    return dict(depth=6, hidden_size=384, num_heads=4)
def DiT_B():
    return dict(depth=12, hidden_size=768, num_heads=12)
def DiT_L():
    return dict(depth=24, hidden_size=1024, num_heads=16)

# Model size
DiT_configs = {'DiT-S': DiT_S(), 'DiT-B': DiT_B(), 'DiT-L': DiT_L()}


@register_custom_model
class ActionModelConfig(PretrainedConfig):
    def __init__(
        self,
        action_dim: int = 7,
        model_type: str = 'DiT-B',
        future_action_window_size: int = 15,
        past_action_window_size: int = 0,
        repeated_diffusion_steps: int = 4,
        noise_schedule: str = 'squaredcos_cap_v2',
        diffusion_steps: int = 100,
        clip_model_name: str = 'openai/clip-vit-large-patch14',
        dinov2_model_name: str = 'vit_base_patch14_reg4_dinov2.lvd142m',
        img_size: int = 224,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        qformer_tokens: int = 32,
        **kwargs,
    ):
        self.action_dim = action_dim
        self.model_type = model_type
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.noise_schedule = noise_schedule
        self.diffusion_steps = diffusion_steps
        self.clip_model_name = clip_model_name
        self.dinov2_model_name = dinov2_model_name
        self.img_size = img_size
        self.norm_stats = norm_stats
        self.qformer_tokens = qformer_tokens
        super().__init__(**kwargs)


# Create ActionModel
class ActionModel(PreTrainedModel):
    config_class = ActionModelConfig

    def __init__(self, config: ActionModelConfig):
        super().__init__(config)
        self.in_channels = config.action_dim
        self.noise_schedule = config.noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = config.diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", noise_schedule=self.noise_schedule, diffusion_steps=self.diffusion_steps, sigma_small=True, learn_sigma=False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False

        # load DiT config
        dit_config = DiT_configs[config.model_type]

        # load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(self.config.clip_model_name)
        # load image encoder
        self.image_encoder = create_model(
            self.config.dinov2_model_name, pretrained=True, img_size=self.config.img_size, drop_path_rate=0.1, proj_drop_rate=0.1
        )
        self.qformer = QFormer(
            num_queries=self.config.qformer_tokens,
            embed_dim=self.image_encoder.embed_dim,
            cross_dim=self.text_encoder.config.hidden_size,
            num_heads=self.image_encoder.embed_dim // 64,
            dropout_rate=0.1,
            with_film=True,
        )

        self.past_action_window_size = config.past_action_window_size
        self.future_action_window_size = config.future_action_window_size
        self.net = DiT(
            text_emb_dim=self.text_encoder.config.hidden_size,
            img_emb_dim=self.image_encoder.embed_dim,
            in_channels=self.in_channels,
            class_dropout_prob=0.1,
            learn_sigma=learn_sigma,
            future_action_window_size=self.future_action_window_size,
            past_action_window_size=self.past_action_window_size,
            num_image_tokens=self.config.qformer_tokens,
            **dit_config
        )

    def forward(
        self, 
        input_ids: torch.Tensor,            # [bs, 77]
        attention_mask: torch.Tensor,       # [bs, 77]
        image_inputs: torch.Tensor,         # [bs, 3, 224, 224]
        actions: torch.Tensor,              # [bs, T, C]
        action_masks: torch.Tensor,         # [bs, T]
    ):
        # repeat inputs for training multiple steps
        input_ids = input_ids.repeat(self.config.repeated_diffusion_steps, 1)
        attention_mask = attention_mask.repeat(self.config.repeated_diffusion_steps, 1)
        image_inputs = image_inputs.repeat(self.config.repeated_diffusion_steps, 1, 1, 1)
        actions = actions.repeat(self.config.repeated_diffusion_steps, 1, 1)
        action_masks = action_masks.repeat(self.config.repeated_diffusion_steps, 1)

        # sample random noise and timestep
        noise = torch.randn_like(actions)  # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (actions.size(0),), device=actions.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(actions, timestep, noise).to(dtype=actions.dtype)

        # get text and image features
        with torch.no_grad():
            text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state  # [bs, 77, 768]
        image_features = self.image_encoder.forward_features(image_inputs)  # [bs, 261, 768]
        image_features = self.qformer(image_features, text_features)  # [bs, 32, 768]

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, text_features, image_features)

        assert noise_pred.shape == noise.shape == actions.shape
        # Compute L2 loss
        batch_loss = ((noise_pred.float() - noise.float()) ** 2).mean(-1)  # [B, T]
        masked_loss = batch_loss * action_masks
        loss = masked_loss.sum(dim=1) / action_masks.sum(dim=1).clamp(min=1)  # [B]
        loss = loss.mean()

        return (loss,)

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion=create_diffusion(
            timestep_respacing="ddim"+str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion
