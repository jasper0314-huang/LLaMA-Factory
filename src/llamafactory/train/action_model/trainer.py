from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, SaveLastCheckpointMixin


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from ...hparams import FinetuningArguments


class ActionModelTrainer(Trainer, SaveLastCheckpointMixin):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()
