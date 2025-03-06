from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, List
from PIL import Image

import torch
import timm
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from ...data import SFTDataCollatorWith4DAttentionMask
from ...data.processor.supervised import SupervisedDatasetProcessor
from ...extras.logging import get_logger
from ...hparams import DataArguments, FinetuningArguments

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...data.template import Template


logger = get_logger(__name__)


@dataclass
class ActionModelTransform:
    r"""
    Convert rlds data into ActionModel inputs.
    """
    finetuning_args: "FinetuningArguments"

    def __post_init__(self):
        # text tokenizer
        self.clip_tokenizer = AutoTokenizer.from_pretrained(self.finetuning_args.clip_name)
        # image transform
        img_size = self.finetuning_args.default_image_resolution
        self.dinov2_transform = transforms.Compose([
            transforms.Resize(size=img_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # For future action predictions
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"].squeeze(0))
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        action = torch.tensor(action, dtype=torch.float32)
        if "action_mask" in rlds_batch:
            action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)
        else:
            action_mask = None

        # prepare action model text and image inputs
        text_inputs = self.clip_tokenizer(text=lang, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        image_inputs = self.dinov2_transform(img)

        return dict(
            input_ids=text_inputs.input_ids.squeeze(0),
            attention_mask=text_inputs.attention_mask.squeeze(0),
            image_inputs=image_inputs,
            actions=action,
            action_masks=action_mask,
        )
