from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Dict, Any, Callable
from PIL import Image

import torch
from transformers import DataCollatorWithPadding

from ...data import SFTDataCollatorWith4DAttentionMask
from ...data.processor.unsupervised import UnsupervisedDatasetProcessor
from ...extras.logging import get_logger
from ...hparams import DataArguments
from ..action_model.data_utils import ActionModelTransform

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...data.template import Template


logger = get_logger(__name__)


@dataclass
class ReasoningVLARLDSBatchTransform:
    r"""
    Convert rlds data into VLMs inputs compatible with LLaMA-Factory.

    Refer to `RLDSBatchTransform` and `_encode_data_example`.
    """
    vlm_template: "Template"
    vlm_tokenizer: "PreTrainedTokenizer"
    vlm_processor: "ProcessorMixin"
    action_tokenizer: "PreTrainedTokenizer"
    action_img_transform: "Callable"
    data_args: "DataArguments"
    reasoning_prompt_template: str
    instruction_placeholder: str

    def __post_init__(self):
        self.vlm_data_processor = UnsupervisedDatasetProcessor(
            template=self.vlm_template,
            tokenizer=self.vlm_tokenizer,
            processor=self.vlm_processor,
            data_args=self.data_args,
        )
        self.action_transform = ActionModelTransform(
            tokenizer=self.action_tokenizer,
            transform=self.action_img_transform,
        )

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # For future action predictions
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        vlm_inputs = self._prepare_vlm_inputs(img, lang)
        action_inputs = self.action_transform(rlds_batch)
        action = action_inputs.pop("actions")
        action_mask = action_inputs.pop("action_masks")

        return {
            "vlm_inputs": vlm_inputs,
            "action_inputs": action_inputs,
            "actions": action,
            "action_masks": action_mask,
        }

    def _prepare_vlm_inputs(self, img: Image, lang: str) -> Dict[str, Any]:
        # follow the format of _encode_data_example
        assert self.instruction_placeholder in self.reasoning_prompt_template
        reasoning_prompt = self.reasoning_prompt_template.replace(self.instruction_placeholder, lang)
        prompt = {
            "role": "user",
            "content": reasoning_prompt,
        }
        images, videos, audios = [img], [], []

        input_ids, labels = self.vlm_data_processor._encode_data_example(
            prompt=[prompt],
            response=[],
            system='',
            tools='',
            images=images,
            videos=videos,
            audios=audios,
        )
        attention_mask = [1] * len(input_ids)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            videos=videos,
        )

    def _prepare_action_inputs(self, img: Image, lang: str) -> Dict[str, Any]:
        return {}


@dataclass
class ReasoningVLADataCollator(SFTDataCollatorWith4DAttentionMask):
    r"""
    Data collator for VLA finetuning.

    Features should contain input_ids, attention_mask, labels, actions, action_masks, and optionally contain images and videos.
    Refer to `PaddedCollatorForActionPrediction`.
    """
    action_tokenizer: "PreTrainedTokenizer" = None

    def __post_init__(self):
        super().__post_init__()
        self.action_collator = DataCollatorWithPadding(tokenizer=self.action_tokenizer)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # vlm
        vlm_features = [feature.pop("vlm_inputs") for feature in features]
        vlm_inputs = super().__call__(vlm_features)

        # action model
        action_features = [feature.pop("action_inputs") for feature in features]
        action_inputs = self.action_collator(action_features)

        # adding continuous actions and batch processing.
        actions = [feature.pop("actions") for feature in features]
        actions = torch.stack(actions)
        action_masks = [feature.pop("action_masks") for feature in features]
        action_masks = torch.stack(action_masks)

        return {
            "vlm_inputs": vlm_inputs,
            "action_inputs": action_inputs,
            "actions": actions,
            "action_masks": action_masks,
        }
