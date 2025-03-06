import json
from typing import TYPE_CHECKING, List, Optional, Dict
from transformers import TrainerCallback
from pathlib import Path

import torch.distributed as dist

from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...extras.misc import count_parameters
from ..trainer_utils import create_modelcard_and_push
from ..callbacks import SaveLastCheckpointCallback

from .trainer import ActionModelTrainer
from .data_utils import ActionModelTransform
from .models.action_model import ActionModelConfig, ActionModel

from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def run_action_model_ft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    assert training_args.do_train, "Only support training now."
    batch_transform = ActionModelTransform(
        finetuning_args=finetuning_args,
        **ActionModel.get_tokenizer_and_image_transform(
            finetuning_args.clip_name,
            finetuning_args.dinov2_name,
            finetuning_args.default_image_resolution,
        ),
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
        num_read_threads=finetuning_args.num_read_threads,
        num_transform_threads=finetuning_args.num_transform_threads,
    )

    # save the dataset statistics
    run_dir = Path(training_args.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    if dist.is_initialized():
        dist.barrier()

    # load model
    with open(run_dir / "dataset_statistics.json", "r") as f:
        norm_stats = json.load(f)
    model = load_model(model_args, finetuning_args, norm_stats)

    # Initialize our Trainer
    trainer = ActionModelTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
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

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)


def load_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
):
    # load model
    config = ActionModelConfig(
        action_dim=finetuning_args.action_dim,
        model_type=finetuning_args.action_model_type,
        future_action_window_size=finetuning_args.future_action_window_size,
        past_action_window_size=finetuning_args.past_action_window_size,
        repeated_diffusion_steps=finetuning_args.repeated_diffusion_steps,
        img_size=finetuning_args.default_image_resolution,
        norm_stats=norm_stats,
    )
    if model_args.train_from_scratch:
        model = ActionModel(config)
    else:
        model = ActionModel.from_pretrained(
            config=config,
            pretrained_model_name_or_path=model_args.model_name_or_path,
        )
    model.train()
    model.text_encoder.eval()

    # set trainable parameters
    model.text_encoder.requires_grad_(False)

    # print trainable parameters
    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    logger.info_rank0(param_stats)

    return model
