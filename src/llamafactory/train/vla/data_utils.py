from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Dict, Any
from PIL import Image

import torch

from ...data import SFTDataCollatorWith4DAttentionMask
from ...data.processor.supervised import SupervisedDatasetProcessor
from ...extras.logging import get_logger
from ...hparams import DataArguments

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...data.template import Template


logger = get_logger(__name__)


@dataclass
class VLARLDSBatchTransform:
    r"""
    Convert rlds data into VLMs inputs compatible with LLaMA-Factory.

    Refer to `RLDSBatchTransform` and `_encode_data_example`.
    """
    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: "ProcessorMixin"
    data_args: "DataArguments"

    def __post_init__(self):
        self.data_processor = SupervisedDatasetProcessor(
            template=self.template,
            tokenizer=self.tokenizer,
            processor=self.processor,
            data_args=self.data_args,
        )

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # For future action predictions
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # follow the format of _encode_data_example
        prompt = {
            "role": "user",
            "content": f"<image>What action should the robot take to {lang}?",
        }
        response = {
            "role": "assistant",
            "content": "",  # we don't add the action to the chat answer
        }
        images, videos, audios = [img], [], []

        input_ids, labels = self.data_processor._encode_data_example(
            prompt=[prompt],
            response=[response],
            system='',
            tools='',
            images=images,
            videos=videos,
            audios=audios,
        )
        attention_mask = [1] * len(input_ids)

        action = torch.tensor(action, dtype=torch.float32)
        if "action_mask" in rlds_batch:
            action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)
        else:
            action_mask = None

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images,
            videos=videos,
            actions=action,
            action_masks=action_mask,
        )


@dataclass
class VLADataCollator(SFTDataCollatorWith4DAttentionMask):
    r"""
    Data collator for VLA finetuning.

    Features should contain input_ids, attention_mask, labels, actions, action_masks, and optionally contain images and videos.
    Refer to `PaddedCollatorForActionPrediction`.
    """
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # adding continuous actions and batch processing.
        actions = [feature.pop("actions") for feature in features]
        actions = torch.stack(actions)
        action_masks = [feature.pop("action_masks") for feature in features]
        action_masks = torch.stack(action_masks)

        # use the multimodal collator for the rest of the features
        ret = super().__call__(features)
        ret["actions"] = actions
        ret["action_masks"] = action_masks
        return ret
