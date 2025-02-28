from typing_extensions import override
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLCausalLMOutputWithPast
import torch
from ....extras import logging
from ....model.loader import register_custom_model
from .action_model.action_model import ActionModel
from .base import Base_VLAConfig, Base_VLAOutput

logger = logging.get_logger(__name__)


@dataclass
class Qwen2_VL_VLAOutput(Base_VLAOutput, Qwen2VLCausalLMOutputWithPast):
    """
    Output for Qwen2-VL VLA model.
    """


class Qwen2_VL_VLAConfig(Base_VLAConfig, Qwen2VLConfig):
    """
    Config for Qwen2-VL VLA model.
    """


@register_custom_model
class Qwen2_VL_VLA(Qwen2VLForConditionalGeneration):
    config_class = Qwen2_VL_VLAConfig

    def __init__(self, config: Qwen2_VL_VLAConfig):
        super().__init__(config)
        # action model
        self.action_model = ActionModel(
            model_type=self.config.action_model_type,
            token_size=self.config.hidden_size,
            in_channels=self.config.action_dim,
            future_action_window_size=self.config.future_action_window_size,
            past_action_window_size=self.config.past_action_window_size,
        )

    @override
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_VL_VLAOutput]:
        r"""
        Override forward to compute next frame prediction loss.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # extract the cognition feature here
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, hidden_states.size(-1))  # [B, D]
        cognition_features = hidden_states.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        loss = None
        if actions is not None:
            actions_history = actions[:,0:self.config.past_action_window_size,:]
            actions_future = actions[:, -(self.config.future_action_window_size+1):, :]

            # repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
            actions_repeated = actions_future.repeat(self.config.repeated_diffusion_steps, 1, 1)
            actions_history_repeated = actions_history.repeat(self.config.repeated_diffusion_steps, 1, 1)
            cognition_features_repeated = cognition_features.repeat(self.config.repeated_diffusion_steps, 1, 1) # [repeated_diffusion_steps*B, 1, D]

            # calculate action loss
            loss = self.action_model.loss(actions_repeated, cognition_features_repeated)

        if not return_dict:
            output = (cognition_features,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_VL_VLAOutput(
            loss=loss,
            cognition_features=cognition_features,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    @torch.inference_mode()
    def predict_action(
        self,
        image: Image, 
        instruction: str, 
        unnorm_key: Optional[str] = None, 
        cfg_scale: float = 1.5, 
        use_ddim: bool = False,
        num_ddim_steps: int = 5,
        **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cfg_scale: Scaling factor for classifier-free guidance (CFG); if == 1.0, CFG is disabled.
        @param use_ddim: Use DDIM sampling instead of DDPM sampling.
        @param num_ddim_steps: Number of DDIM steps to use for sampling.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        if not (
            hasattr(self, "template")
            and hasattr(self, "processor")
            and hasattr(self, "tokenizer")
            and hasattr(self, "collator")
        ):
            raise ValueError("template, processor, tokenizer, and collator must be set using `setattr` before calling `predict_action`")

        # build inputs
        messages = [
            {
                "role": "user",
                "content": f"<image>What action should the robot take to {instruction.lower()}?"
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
        images, videos = [image], []
        messages = self.template.mm_plugin.process_messages(messages, images, videos, self.processor)
        input_ids, _ = self.template.encode_oneturn(self.tokenizer, messages, system=None, tools=None)
        input_ids, _ = self.template.mm_plugin.process_token_ids(input_ids, None, images, videos, self.tokenizer, self.processor)

        features = self.collator([{
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "images": images,
            "videos": videos,
        }])

        # convert features to device
        features = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }
        cognition_features = self.forward(**features).cognition_features  # [B, 1, D]
        assert cognition_features.shape[0] == 1, "Batch size must be 1 for action prediction"
        B = cognition_features.shape[0]

        using_cfg = cfg_scale > 1.0
        model_dtype = self.dtype

        # Sample random noise
        noise = torch.randn(B, self.config.future_action_window_size+1, self.action_model.in_channels, device=cognition_features.device).to(model_dtype)  #[B, T, D]
    
        # Setup classifier-free guidance:
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.action_model.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([cognition_features, uncondition], 0)
            cfg_scale = cfg_scale
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=cognition_features)
            sample_fn = self.action_model.net.forward

        # DDIM Sampling
        if use_ddim and num_ddim_steps is not None:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(sample_fn, 
                                                                noise.shape, 
                                                                noise, 
                                                                clip_denoised=False,
                                                                model_kwargs=model_kwargs,
                                                                progress=False,
                                                                device=cognition_features.device,
                                                                eta=0.0
                                                                )
        else:
            # DDPM Sampling
            samples = self.action_model.diffusion.p_sample_loop(sample_fn, 
                                                                    noise.shape, 
                                                                    noise, 
                                                                    clip_denoised=False,
                                                                    model_kwargs=model_kwargs,
                                                                    progress=False,
                                                                    device=cognition_features.device
                                                                    )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        normalized_actions = samples[0].cpu().numpy()

        # Un-normalize Actions        
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, normalized_actions

    def get_action_stats(self, unnorm_key=None):
        """Get the action normalization statistics."""
        unnorm_key = self._check_unnorm_key(self.config.norm_stats, unnorm_key)
        return self.config.norm_stats[unnorm_key]["action"]

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
