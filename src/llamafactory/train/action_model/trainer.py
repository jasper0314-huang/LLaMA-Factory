from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled
from typing_extensions import override

from ...extras import logging
from ..trainer_utils import create_custom_scheduler, SaveLastCheckpointMixin


logger = logging.get_logger(__name__)


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp # type: ignore


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
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameter_names = self.get_decay_parameter_names(opt_model)

            decay_main_params = [
                p for n, p in opt_model.named_parameters()
                if (
                    p.requires_grad
                    and n in decay_parameter_names
                    and "image_encoder" not in n
                )
            ]
            no_decay_main_params = [
                p for n, p in opt_model.named_parameters()
                if (
                    p.requires_grad
                    and n not in decay_parameter_names
                    and "image_encoder" not in n
                )
            ]
            decay_image_encoder_params = [
                p for n, p in opt_model.named_parameters()
                if (
                    p.requires_grad
                    and n in decay_parameter_names
                    and "image_encoder" in n
                )
            ]
            no_decay_image_encoder_params = [
                p for n, p in opt_model.named_parameters()
                if (
                    p.requires_grad
                    and n not in decay_parameter_names
                    and "image_encoder" in n
                )
            ]

            image_encoder_lr_config = {"lr": self.finetuning_args.image_encoder_lr} if self.finetuning_args.image_encoder_lr is not None else {}

            # param groups
            optimizer_grouped_parameters = [
                {
                    "params": decay_main_params,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": no_decay_main_params,
                    "weight_decay": 0.0,
                },
                {
                    "params": decay_image_encoder_params,
                    "weight_decay": self.args.weight_decay,
                    **image_encoder_lr_config,
                },
                {
                    "params": no_decay_image_encoder_params,
                    "weight_decay": 0.0,
                    **image_encoder_lr_config,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes # type: ignore

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

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
