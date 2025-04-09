import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import random
import subprocess
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from typing_extensions import override

from llamafactory.extras import logging
from llamafactory.train.callbacks import LogCallback, PissaConvertCallback, ReporterCallback, WandbTimerCallback
from llamafactory.hparams import get_train_args
from llamafactory.train.trainer_utils import get_swanlab_callback
from llamafactory.extras.misc import get_device_count


if TYPE_CHECKING:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


logger = logging.get_logger(__name__)


class WandbReporterCallback4VLA(ReporterCallback):
    r"""
    A callback for reporting training status to wandb.
    To avoid precision mismatch, we discard the norm_stats.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return
        if "wandb" in args.report_to:
            import wandb
            model_args_dict = self.model_args.to_dict()
            model_args_dict["additional_model_args"].pop("norm_stats")
            wandb.config.update(
                {
                    "model_args": model_args_dict,
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


def run(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    callbacks.append(WandbTimerCallback())
    assert not dist.is_initialized(), "Distributed groups should only be initialized with the TrainingArguments!!"
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    callbacks.append(WandbReporterCallback4VLA(model_args, data_args, finetuning_args, generating_args))  # add to last

    from llamafactory.train.reasoning_vla import run_reasoning_vla_ft  # import here to avoid dist initialized by overwatch
    assert finetuning_args.stage == "sft"
    run_reasoning_vla_ft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)


if __name__ == "__main__":
    force_torchrun = os.getenv("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
    if (
        (force_torchrun or get_device_count() > 1)
        and not os.getenv("TORCHRUN_REENTRY_FLAG", "0") == "1"
    ):
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
        logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
        env = os.environ.copy()
        env["TORCHRUN_REENTRY_FLAG"] = "1"
        process = subprocess.run([
            "torchrun",
            "--nnodes", os.getenv("NNODES", "1"),
            "--node_rank", os.getenv("NODE_RANK", "0"),
            "--nproc_per_node", os.getenv("NPROC_PER_NODE", str(get_device_count())),
            "--master_addr", master_addr,
            "--master_port", master_port,
            __file__,
        ] + sys.argv[1:], env=env)
        sys.exit(process.returncode)
    else:
        run()