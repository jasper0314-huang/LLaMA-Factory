from typing_extensions import override
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np
import torch

from transformers import PretrainedConfig, PreTrainedModel, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
from ...action_model.models import ActionModel
from ....extras import logging
from ....model.model_utils.visual import get_image_seqlen, get_patch_size, get_vision_feature_select_strategy

logger = logging.get_logger(__name__)


@dataclass
class ReasoningVLAOutput(ModelOutput):
    """
    Output of ReasoningVLA.
    """


class ReasoningVLAConfig(PretrainedConfig):
    def __init__(
        self,
        reasoning_vlm_name_or_path: str = None,
        action_model_name_or_path: str = None,
        vlm_image_max_pixels: int = 262144,  # 512 * 512
        vlm_image_min_pixels: int = 65536,  # 256 * 256
        vlm_video_max_pixels: int = 65536,  # 256 * 256
        vlm_video_min_pixels: int = 16384,  # 128 * 128
        vlm_video_fps: int = 2,
        vlm_video_maxlen: int = 16,
        max_response_length: int = 2048,
        vlm_attn_implementation: str = "auto",
        vlm_compute_dtype: torch.dtype = None,
        generating_args: Dict[str, Any] = None,
        max_vlm_embeddings: int = 256,
        **kwargs,
    ):
        self.reasoning_vlm_name_or_path = reasoning_vlm_name_or_path
        self.action_model_name_or_path = action_model_name_or_path
        self.vlm_image_max_pixels = vlm_image_max_pixels
        self.vlm_image_min_pixels = vlm_image_min_pixels
        self.vlm_video_max_pixels = vlm_video_max_pixels
        self.vlm_video_min_pixels = vlm_video_min_pixels
        self.vlm_video_fps = vlm_video_fps
        self.vlm_video_maxlen = vlm_video_maxlen
        self.max_response_length = max_response_length
        self.vlm_attn_implementation = vlm_attn_implementation
        self.vlm_compute_dtype = vlm_compute_dtype
        self.generating_args = generating_args
        self.max_vlm_embeddings = max_vlm_embeddings
        super().__init__(**kwargs)


class ReasoningVLA(PreTrainedModel):
    config_class = ReasoningVLAConfig

    def __init__(self, config: ReasoningVLAConfig):
        super().__init__(config)
        self.reasoning_vlm = AutoModelForVision2Seq.from_pretrained(
            config.reasoning_vlm_name_or_path,
            torch_dtype=config.vlm_compute_dtype,
            attn_implementation=config.vlm_attn_implementation,
        )
        self.action_model = ActionModel.from_pretrained(
            config.action_model_name_or_path,
        )
        self.generating_args = config.generating_args

        # get processor and tokenizer
        self.vlm_tokenizer, self.vlm_processor = self._init_vlm_processor()
        self.action_tokenizer, self.action_img_transform = self.action_model.get_tokenizer_and_image_transform(
            self.action_model.config.clip_model_name,
            self.action_model.config.dinov2_model_name,
            self.action_model.config.img_size,
            ret_dict=False,
        )

    @override
    def forward(
        self,
        vlm_inputs: Dict[str, Any],
        action_inputs: Dict[str, Any],
        actions: Optional[torch.FloatTensor] = None,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, ReasoningVLAOutput]:
        with torch.no_grad():
            full_sequences = self.generate_sequence(vlm_inputs)
        # pop up old inputs that will not be used
        vlm_inputs.pop("input_ids")
        vlm_inputs.pop("attention_mask")
        vlm_inputs.pop("labels")
        # forward reasoning vlm with full sequences
        vlm_last_hidden_states = self.reasoning_vlm(
            full_sequences,
            **vlm_inputs,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        predicted_embeddings, pad_masks = self.extract_predicted_embeddings(full_sequences, vlm_last_hidden_states)
        # action model forward with predicted embeddings
        loss = self.action_model(**action_inputs, actions=actions, action_masks=action_masks, addit_cond_embeddings=predicted_embeddings)[0]
        return (loss,)

    def generate_sequence(self, vlm_inputs: Dict[str, Any]):
        outputs = self.reasoning_vlm.generate(
            **vlm_inputs,
            **self.generating_args,
            use_cache=True,
        )
        return outputs

    def extract_predicted_embeddings(self, full_sequences: torch.LongTensor, vlm_last_hidden_states: torch.FloatTensor):
        # convert full_sequences to text and extract the answer
        full_texts = self.vlm_tokenizer.batch_decode(full_sequences, skip_special_tokens=False)
        results = []
        pad_masks = []
        for idx, full_text in enumerate(full_texts):
            answer = extract_boxed_content(full_text)
            if answer is None:
                predicted_embeddings = vlm_last_hidden_states[idx, -self.config.max_vlm_embeddings:]
                # mask whole embeddings
                pad_mask = torch.zeros(predicted_embeddings.shape[0], device=predicted_embeddings.device)
            else:
                answer_indices = find_substring_token_indices(full_text, answer, self.vlm_tokenizer)
                answer_indices = answer_indices.to(vlm_last_hidden_states.device)
                predicted_embeddings = vlm_last_hidden_states[idx, answer_indices]
                pad_mask = torch.ones(predicted_embeddings.shape[0], device=predicted_embeddings.device)
            # pad or truncate predicted_embeddings to max_vlm_embeddings
            predicted_embeddings, pad_mask = pad_or_truncate(predicted_embeddings, pad_mask, self.config.max_vlm_embeddings)
            results.append(predicted_embeddings)
            pad_masks.append(pad_mask)
        results = torch.stack(results)
        pad_masks = torch.stack(pad_masks)
        return results, pad_masks

    def get_processors(self):
        return dict(
            vlm_tokenizer=self.vlm_tokenizer,
            vlm_processor=self.vlm_processor,
            action_tokenizer=self.action_tokenizer,
            action_img_transform=self.action_img_transform,
        )

    def _init_vlm_processor(self):
        # load reasoning vlm processor and tokenizer
        vlm_tokenizer = AutoTokenizer.from_pretrained(
            self.config.reasoning_vlm_name_or_path,
            padding_side="left",  # important for using `generate`
        )
        vlm_processor = AutoProcessor.from_pretrained(self.config.reasoning_vlm_name_or_path)
        self._patch_vlm_processor(vlm_processor, vlm_tokenizer)
        return vlm_tokenizer, vlm_processor

    def _patch_vlm_processor(self, vlm_processor: AutoProcessor, vlm_tokenizer: AutoTokenizer):
        # follow llamafactory.model.patcher.patch_processor
        setattr(vlm_processor, "tokenizer", vlm_tokenizer)
        setattr(vlm_processor, "image_seqlen", get_image_seqlen(self.reasoning_vlm.config))
        setattr(vlm_processor, "patch_size", get_patch_size(self.reasoning_vlm.config, vlm_processor))
        setattr(vlm_processor, "image_max_pixels", self.config.vlm_image_max_pixels)
        setattr(vlm_processor, "image_min_pixels", self.config.vlm_image_min_pixels)
        setattr(vlm_processor, "video_max_pixels", self.config.vlm_video_max_pixels)
        setattr(vlm_processor, "video_min_pixels", self.config.vlm_video_min_pixels)
        setattr(vlm_processor, "video_fps", self.config.vlm_video_fps)
        setattr(vlm_processor, "video_maxlen", self.config.vlm_video_maxlen)
        setattr(vlm_processor, "vision_feature_select_strategy", get_vision_feature_select_strategy(self.reasoning_vlm.config, vlm_processor))


def pad_or_truncate(tensor: torch.Tensor, pad_mask: torch.Tensor, target_len: int, pad_value: float = 0.0):
    T, D = tensor.shape
    if T == target_len:
        return tensor, pad_mask
    elif T < target_len:
        pad_len = target_len - T
        pad_tensor = torch.full((pad_len, D), pad_value, device=tensor.device, dtype=tensor.dtype)
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        padded_mask = torch.cat([
            pad_mask,
            torch.zeros(pad_len, dtype=torch.long, device=tensor.device)
        ])
        return padded_tensor, padded_mask
    else:
        truncated_tensor = tensor[:target_len]
        truncated_mask = pad_mask[:target_len]
        return truncated_tensor, truncated_mask


def find_substring_token_indices(full_text: str, substr: str, tokenizer: AutoTokenizer):
    char_start = full_text.rfind(substr)
    char_end = char_start + len(substr)

    if char_start == -1:
        raise ValueError(f"The substring ({substr}) was not found in the full text ({full_text}).")

    encoding = tokenizer(full_text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    token_start, token_end = None, None
    for idx, (start, end) in enumerate(offsets):
        if start <= char_start < end:
            token_start = idx
        if start < char_end <= end:
            token_end = idx + 1
            break

    if token_start is not None and token_end is None:
        for idx, (start, end) in enumerate(offsets[token_start:], start=token_start):
            if end >= char_end:
                token_end = idx + 1
                break

    token_indices = torch.arange(token_start, token_end)
    return token_indices


# https://github.com/hiyouga/MathRuler
def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    The system prompt may already contain \\boxed{}, so we need to find the last \\boxed{} and extract the content.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break
    if end_pos != -1:
        content = content[:end_pos].strip()
        if len(content) > 0:
            return content
    return None
