import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import random
import subprocess
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llamafactory.extras import logging
from llamafactory.train.sft import run_sft
from llamafactory.train.callbacks import LogCallback, PissaConvertCallback, ReporterCallback
from llamafactory.hparams import get_train_args
from llamafactory.train.trainer_utils import get_swanlab_callback
from llamafactory.extras.misc import get_device_count


if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = logging.get_logger(__name__)


def run(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    assert not dist.is_initialized(), "Distributed groups should only be initialized with the TrainingArguments!!"
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())

    if finetuning_args.use_swanlab:
        callbacks.append(get_swanlab_callback(finetuning_args))

    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))  # add to last

    assert finetuning_args.stage == "sft"
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)


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