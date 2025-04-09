from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, List, Callable
from PIL import Image

import torch
from ...extras.logging import get_logger
from ...hparams import FinetuningArguments

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


logger = get_logger(__name__)


@dataclass
class ActionModelTransform:
    r"""
    Convert rlds data into ActionModel inputs.
    """
    tokenizer: "PreTrainedTokenizer"
    transform: "Callable"

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        # For future action predictions
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]

        img = Image.fromarray(rlds_batch["observation"]["image_primary"].squeeze(0))
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        action = torch.tensor(action, dtype=torch.float32)
        if "action_mask" in rlds_batch:
            action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)
        else:
            action_mask = torch.ones(action.shape[:-1], dtype=torch.bool)

        # prepare action model text and image inputs
        text_inputs = self.tokenizer(text=lang, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        image_inputs = self.transform(img)

        return dict(
            input_ids=text_inputs.input_ids.squeeze(0),
            attention_mask=text_inputs.attention_mask.squeeze(0),
            image_inputs=image_inputs,
            actions=action,
            action_masks=action_mask,
        )
