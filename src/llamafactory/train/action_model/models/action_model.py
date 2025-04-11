import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.clip.modeling_clip import CLIPTextModel
from timm.models import create_model
from diffusers import DDPMScheduler, DDIMScheduler

from .qformer import QFormer, SinusoidalPosEmb


class SEGMENT_TYPE(Enum):
    IMAGE = 0
    TEXT = 1


class ActionModelConfig(PretrainedConfig):
    def __init__(
        self,
        action_dim: int = 7,
        num_actions_chunk: int = 16,
        qformer_tokens: int = 32,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        repeated_diffusion_steps: int = 4,
        noise_schedule: str = 'squaredcos_cap_v2',
        diffusion_steps: int = 1000,
        clip_model_name: str = 'openai/clip-vit-large-patch14',
        dinov2_model_name: str = 'vit_base_patch14_reg4_dinov2.lvd142m',
        img_size: int = 224,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        noise_prediction_type: str = 'epsilon',
        use_segment_embeddings: bool = False,
        **kwargs,
    ):
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.qformer_tokens = qformer_tokens
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.noise_schedule = noise_schedule
        self.diffusion_steps = diffusion_steps
        self.clip_model_name = clip_model_name
        self.dinov2_model_name = dinov2_model_name
        self.img_size = img_size
        self.norm_stats = norm_stats
        self.noise_prediction_type = noise_prediction_type
        self.use_segment_embeddings = use_segment_embeddings
        super().__init__(**kwargs)


# Create ActionModel
class ActionModel(PreTrainedModel):
    config_class = ActionModelConfig

    def __init__(self, config: ActionModelConfig):
        super().__init__(config)
        self.norm_stats = config.norm_stats

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.diffusion_steps,
            beta_schedule=config.noise_schedule,
            prediction_type=config.noise_prediction_type,
        )
        self.noise_scheduler_eval = DDIMScheduler(
            num_train_timesteps=config.diffusion_steps,
            beta_schedule=config.noise_schedule,
            prediction_type=config.noise_prediction_type,
        )

        # load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(self.config.clip_model_name)
        # make sure text_encoder always in eval mode
        self.text_encoder.eval()
        self.text_encoder.train = lambda mode: self.text_encoder

        # load image encoder
        self.image_encoder = create_model(
            self.config.dinov2_model_name, pretrained=True, img_size=self.config.img_size, drop_path_rate=0.1, proj_drop_rate=0.1
        )
        self.qformer = QFormer(
            num_queries=self.config.qformer_tokens,
            embed_dim=self.image_encoder.embed_dim,
            num_heads=self.image_encoder.embed_dim // 64,
            mlp_ratio=4,
            qkv_bias=False,
            norm_layer=nn.LayerNorm,
            dropout_rate=0.0,
            drop_path=0.0,
            use_checkpoint=False,
            with_film=True,
            cross_dim=self.text_encoder.config.hidden_size,
        )

        self.action_dim = config.action_dim
        self.num_actions_chunk = config.num_actions_chunk
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.transformer = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=self.action_dim,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.hidden_size // 64,
                attention_dropout=0.1,
                _flash_attn_2_enabled= True,
            )
        )
        del self.transformer.model.embed_tokens  # will not be used
        self.time_emb = SinusoidalPosEmb(self.hidden_size)

        self.use_segment_embeddings = config.use_segment_embeddings
        if self.use_segment_embeddings:
            self.segment_embeddings = nn.Embedding(2, self.hidden_size)

        # get tokenizer and image transform
        self.tokenizer, self.image_transform = self.get_tokenizer_and_image_transform(
            self.config.clip_model_name, self.config.dinov2_model_name, self.config.img_size, ret_dict=False
        )

    def forward(
        self, 
        input_ids: torch.Tensor,                        # [B, 77]
        attention_mask: torch.Tensor,                   # [B, 77]
        image_inputs: torch.Tensor,                     # [B, 3, 224, 224]
        actions: torch.Tensor,                          # [B, T, C]
        action_masks: torch.Tensor,                     # [B, T]
        addition_cond_embs: torch.Tensor = None,        # [B, L, 768]
    ):
        # repeat inputs for training multiple steps
        input_ids = input_ids.repeat(self.config.repeated_diffusion_steps, 1)
        attention_mask = attention_mask.repeat(self.config.repeated_diffusion_steps, 1)
        image_inputs = image_inputs.repeat(self.config.repeated_diffusion_steps, 1, 1, 1)
        actions = actions.repeat(self.config.repeated_diffusion_steps, 1, 1)
        action_masks = action_masks.repeat(self.config.repeated_diffusion_steps, 1)
        if addition_cond_embs is not None:
            addition_cond_embs = addition_cond_embs.repeat(self.config.repeated_diffusion_steps, 1, 1)

        # sample random noise and timestep
        noise = torch.randn_like(actions)  # [B', T, C]
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (actions.size(0),), device=actions.device)

        # sample x_t from x
        x_t = self.noise_scheduler.add_noise(actions, noise, timestep)

        # predict noise from x_t
        pred = self.model_forward(x_t, timestep, input_ids, attention_mask, image_inputs, addition_cond_embs)

        # decide the denoising target
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = actions
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(actions, noise, timestep)
        else:
            raise ValueError(f"Unsupported prediction type {self.noise_scheduler.config.prediction_type}")

        assert pred.shape == noise.shape == actions.shape
        # Compute L2 loss
        batch_loss = ((pred.float() - target.float()) ** 2).mean(-1)  # [B', T]
        masked_loss = batch_loss * action_masks  # [B', T]
        loss = masked_loss.sum(dim=1) / action_masks.sum(dim=1).clamp(min=1e-6)  # [B']
        loss = loss.mean()

        return (loss,)

    def model_forward(
        self,
        noised_actions: torch.Tensor,             # [B', T, C]
        timestep: torch.Tensor,                   # [B']
        input_ids: torch.Tensor,                  # [B', 77]
        attention_mask: torch.Tensor,             # [B', 77]
        image_inputs: torch.Tensor,               # [B', 3, 224, 224]
        addition_cond_embs: torch.Tensor = None,  # [B', L, 768]
    ) -> torch.Tensor:  # [B', T, C]
        # text embedding
        with torch.no_grad():
            text_embs = self.text_encoder(input_ids, attention_mask).last_hidden_state  # [B', 77, 768]
        # image embedding
        image_embs = self.image_encoder.forward_features(image_inputs)  # [B', 261, 768]
        image_embs = self.qformer(image_embs, text_embs.mean(dim=1))  # [B', 32, 768]
        # get timestep embedding and pad noised actions to hidden size
        timestep_emb = self.time_emb(timestep).unsqueeze(1)  # [B', 1, 768]
        noised_actions_padded = F.pad(noised_actions, (0, self.hidden_size - noised_actions.shape[2]), mode='constant', value=0)  # [B', T, C] -> [B', T, 768]

        # concat additional condition embeddings
        if addition_cond_embs is not None:
            text_embs = torch.cat([addition_cond_embs, text_embs], dim=1)  # [B', L + 77, 768]

        # add segment embeddings
        if self.use_segment_embeddings:
            text_embs = text_embs + self.segment_embeddings.weight[SEGMENT_TYPE.TEXT.value]  # [B', 77, 768]
            image_embs = image_embs + self.segment_embeddings.weight[SEGMENT_TYPE.IMAGE.value]  # [B', 32, 768]

        # concat all embeddings
        context_embs = torch.cat([text_embs, image_embs, timestep_emb, noised_actions_padded], dim=1)  # [B', 77 + 32 + 1 + T, 768]
        preds = self.transformer(inputs_embeds=context_embs)[0]  # [B', 77 + 32 + 1 + T, 7]

        denoised_actions = preds[:, -noised_actions.shape[1]:, :]  # [B', T, 7]
        return denoised_actions

    @staticmethod
    def get_tokenizer_and_image_transform(clip_model_name: str, dinov2_model_name: str, img_size: int, ret_dict: bool=True):
        transform = transforms.Compose([
            transforms.Resize(size=img_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
        ])
        tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
        if ret_dict:
            return dict(tokenizer=tokenizer, transform=transform)
        else:
            return tokenizer, transform

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image,
        instruction: str,
        num_steps: int = 20,
        ret_unorm_action: bool = True,
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param num_steps: Number of steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        if self.noise_scheduler_eval.num_inference_steps != num_steps:
            self.noise_scheduler_eval.set_timesteps(num_inference_steps=num_steps)

        # Prepare Inputs
        text_inputs = self.tokenizer(text=instruction, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        image_inputs = self.image_transform(image).unsqueeze(0)

        # get and cache dtype and device
        if not hasattr(self, "predict_action_dtype_device"):
            param = next(self.parameters())
            self.predict_action_dtype_device = param.dtype, param.device
        model_dtype, device = self.predict_action_dtype_device

        # Sample random noise
        samples = torch.randn(1, self.num_actions_chunk, self.action_dim, device=device).to(model_dtype)  #[1, T, D]

        # Start sampling
        for t in self.noise_scheduler_eval.timesteps:
            noise_pred = self.model_forward(
                samples,
                t.unsqueeze(0).to(device),
                text_inputs.input_ids.to(device),
                text_inputs.attention_mask.to(device),
                image_inputs.to(device),
            )
            samples = self.noise_scheduler_eval.step(noise_pred, t, samples).prev_sample

        normalized_actions = samples[0].cpu().numpy()
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 

        # Un-normalize Actions
        if ret_unorm_action:
            action_norm_stats = self.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            return actions, normalized_actions
        else:
            return normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
