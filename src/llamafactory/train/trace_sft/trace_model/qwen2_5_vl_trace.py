from typing_extensions import override
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLCausalLMOutputWithPast
import torch
from ....extras import logging
from ....model.loader import register_custom_model
from .base import Base_TraceConfig, Base_TraceOutput

logger = logging.get_logger(__name__)


@dataclass
class Qwen2_5_VL_TraceOutput(Base_TraceOutput, Qwen2_5_VLCausalLMOutputWithPast):
    """
    Output for Qwen2-5 VL Trace model.
    """

class Qwen2_5_VL_TraceConfig(Base_TraceConfig, Qwen2_5_VLConfig):
    """
    Config for Qwen2-5 VL Trace model.
    """


@register_custom_model
class Qwen2_5_VL_Trace(Qwen2_5_VLForConditionalGeneration):
    config_class = Qwen2_5_VL_TraceConfig

    def __init__(self, config: Qwen2_5_VL_TraceConfig):
        super().__init__(config)
        # trace proj
        self.num_trace_points = config.num_trace_points
        self.trace_head = nn.Linear(self.config.hidden_size, 2)  # (x, y)
        self.trace_loss_weight = config.trace_loss_weight

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
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # trace
        trace_labels: Optional[torch.FloatTensor] = None,
        trace_masks: Optional[torch.LongTensor] = None,
        trace_token_indices: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5_VL_TraceOutput]:
        r"""
        Override forward to compute trace loss.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

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
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            bs = shift_logits.shape[0]
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels).view(bs, -1).mean(-1)
            num_non_trace_samples = trace_masks.sum().item()
            loss = loss * trace_masks.float()
            loss = loss.sum() * (1./num_non_trace_samples if num_non_trace_samples > 0 else 0.)

        # predict trace
        expanded_trace_token_indices = trace_token_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[2])  # [bs, num_points, hidden_size]
        trace_hidden_states = hidden_states.gather(1, expanded_trace_token_indices)
        traces = self.trace_head(trace_hidden_states)  # [bs, num_trace_points, 2]

        trace_loss = None
        if trace_labels is not None:
            trace_loss = F.mse_loss(traces.float(), trace_labels.float(), reduction='none')  # [bs, num_trace_points, 2]
            trace_loss = trace_loss.mean(-2).mean(-1)  # [bs,]
            flipped_trace_masks = 1 - trace_masks
            num_trace_samples = flipped_trace_masks.sum().item()
            trace_loss = trace_loss * flipped_trace_masks.float()
            trace_loss = trace_loss.sum() * (1./num_trace_samples if num_trace_samples > 0 else 0.)
            trace_loss = trace_loss * self.trace_loss_weight
            loss = loss + trace_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VL_TraceOutput(
            loss=loss,
            trace_loss=trace_loss,
            logits=logits,
            traces=traces,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
