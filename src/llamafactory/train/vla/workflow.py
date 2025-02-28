import json
from typing import TYPE_CHECKING, List, Optional
from transformers import TrainerCallback
from pathlib import Path

import torch.distributed as dist

from ...data import get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from ..callbacks import SaveLastCheckpointCallback

from .trainer import ReasoningVLATrainer
from .data_utils import VLARLDSBatchTransform, VLADataCollator

from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def run_vla_ft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    batch_transform = VLARLDSBatchTransform(
        template=template,
        data_args=data_args,
        **tokenizer_module,
    )

    assert len(data_args.dataset) == 1, "You can only specify a single OXE mix for training"
    train_dataset = RLDSDataset(
        data_args.dataset_dir,
        data_args.dataset[0],
        batch_transform,
        resize_resolution=finetuning_args.default_image_resolution,
        shuffle_buffer_size=finetuning_args.shuffle_buffer_size,
        future_action_window_size=finetuning_args.future_action_window_size,
        past_action_window_size=finetuning_args.past_action_window_size,
        train=training_args.do_train,
        image_aug=finetuning_args.image_aug,
        load_all_data_for_training=finetuning_args.load_all_data_for_training,
        num_parallel_calls=finetuning_args.num_parallel_calls,
    )

    # save the dataset statistics
    run_dir = Path(training_args.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    if dist.is_initialized():
        dist.barrier()

    # read dataset_statistics as norm_stats
    with open(run_dir / "dataset_statistics.json", "r") as f:
        norm_stats = json.load(f)
    additional_model_args = {
        "action_dim": finetuning_args.action_dim,
        "action_model_type": finetuning_args.action_model_type,
        "future_action_window_size": finetuning_args.future_action_window_size,
        "past_action_window_size": finetuning_args.past_action_window_size,
        "repeated_diffusion_steps": finetuning_args.repeated_diffusion_steps,
        "norm_stats": norm_stats,
    }
    model_args.additional_model_args = additional_model_args
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = VLADataCollator(
        template=template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )

    # Override the training parameters of VLA Trainer
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Initialize our Trainer
    trainer = ReasoningVLATrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        **tokenizer_module,
    )

    if finetuning_args.save_last_ckpt_steps > 0:
        trainer.add_callback(SaveLastCheckpointCallback(trainer, finetuning_args.save_last_ckpt_steps))

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Evaluation
    if training_args.do_eval or training_args.do_predict:
        raise NotImplementedError("do_eval and do_predict are not supported for VLA FT")

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)