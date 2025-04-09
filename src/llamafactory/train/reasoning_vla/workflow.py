import json
from typing import TYPE_CHECKING, List, Optional, Dict
from transformers import TrainerCallback
from pathlib import Path

import torch.distributed as dist

from ...data import get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...extras.misc import count_parameters
from ..trainer_utils import create_modelcard_and_push
from ..callbacks import SaveLastCheckpointCallback

from .trainer import ReasoningVLATrainer
from .data_utils import ReasoningVLARLDSBatchTransform, ReasoningVLADataCollator
from .models import ReasoningVLA, ReasoningVLAConfig

from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments, GeneratingArguments


logger = get_logger(__name__)


def run_reasoning_vla_ft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    assert training_args.do_train, "Only support training now."
    model = load_model(model_args, finetuning_args, generating_args)
    processors = model.get_processors()
    vlm_template = get_template_and_fix_tokenizer(processors["vlm_tokenizer"], data_args)

    batch_transform = ReasoningVLARLDSBatchTransform(
        vlm_template=vlm_template,
        data_args=data_args,
        reasoning_prompt_template=finetuning_args.reasoning_prompt_template,
        instruction_placeholder=finetuning_args.instruction_placeholder,
        **processors,
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

    data_collator = ReasoningVLADataCollator(
        template=vlm_template,
        model=model,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else processors["vlm_tokenizer"].pad_token_id,
        tokenizer=processors["vlm_tokenizer"],
        processor=processors["vlm_processor"],
        action_tokenizer=processors["action_tokenizer"],
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
        tokenizer=processors["vlm_tokenizer"],
        processor=processors["vlm_processor"],
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


flash_attn_mapping = {
    "auto": "auto",
    "disabled": "eager",
    "sdpa": "sdpa",
    "fa2": "flash_attention_2",
}

def load_model(
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
):
    generating_args = generating_args.to_dict()
    generating_args.pop("skip_special_tokens")  # remove from `generate`
    # load model
    config = ReasoningVLAConfig(
        reasoning_vlm_name_or_path=finetuning_args.reasoning_vlm_name_or_path,
        action_model_name_or_path=finetuning_args.action_model_name_or_path,
        vlm_image_max_pixels=model_args.image_max_pixels,
        vlm_image_min_pixels=model_args.image_min_pixels,
        vlm_video_max_pixels=model_args.video_max_pixels,
        vlm_video_min_pixels=model_args.video_min_pixels,
        vlm_video_fps=model_args.video_fps,
        vlm_video_maxlen=model_args.video_maxlen,
        max_response_length=finetuning_args.max_response_length,
        vlm_attn_implementation=flash_attn_mapping[model_args.flash_attn],
        vlm_compute_dtype=model_args.compute_dtype,
        generating_args=generating_args,
    )
    if model_args.train_from_scratch:
        model = ReasoningVLA(config)
    else:
        model = ReasoningVLA.from_pretrained(
            config=config,
            pretrained_model_name_or_path=model_args.model_name_or_path,
        )
    model.train()

    # TODO: set training parameters
    model.reasoning_vlm.requires_grad_(False)
    model.action_model.requires_grad_(True)

    # print trainable parameters
    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    logger.info_rank0(param_stats)

    return model
