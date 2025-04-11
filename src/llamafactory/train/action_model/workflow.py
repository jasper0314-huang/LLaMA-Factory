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
        **ActionModel.get_tokenizer_and_image_transform(
            finetuning_args.clip_model_name,
            finetuning_args.dinov2_model_name,
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
        train=training_args.do_train,
        image_aug=finetuning_args.image_aug,
        num_actions_chunk=finetuning_args.num_actions_chunk,
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
        num_actions_chunk=finetuning_args.num_actions_chunk,
        qformer_tokens=finetuning_args.qformer_tokens,
        num_hidden_layers=finetuning_args.num_hidden_layers,
        hidden_size=finetuning_args.hidden_size,
        intermediate_size=finetuning_args.intermediate_size,
        repeated_diffusion_steps=finetuning_args.repeated_diffusion_steps,
        noise_schedule=finetuning_args.noise_schedule,
        diffusion_steps=finetuning_args.diffusion_steps,
        clip_model_name=finetuning_args.clip_model_name,
        dinov2_model_name=finetuning_args.dinov2_model_name,
        img_size=finetuning_args.default_image_resolution,
        norm_stats=norm_stats,
        noise_prediction_type=finetuning_args.noise_prediction_type,
        use_segment_embeddings=finetuning_args.use_segment_embeddings,
    )
    if model_args.train_from_scratch:
        model = ActionModel(config)

        import os
        import torch
        INIT = os.environ.get('INIT')
        if INIT == "1":
            ckpt = torch.load('/lustre/fs12/portfolios/nvr/users/chipinh/work/research_VLA/Dita/dit_policy_checkpoint.pth')['parameter']
            transformer_ckpt = {k.replace('transformer.', ''): v for k, v in ckpt.items() if k.startswith('transformer.')}
            image_encoder_ckpt = {k.replace('image_tokenizer.tokenizer.', ''): v for k, v in ckpt.items() if k.startswith('image_tokenizer.tokenizer.')}
            qformer_ckpt = {k.replace('image_tokenizer.qformer.', ''): v for k, v in ckpt.items() if k.startswith('image_tokenizer.qformer.')}

            model.transformer.load_state_dict(transformer_ckpt)
            model.image_encoder.load_state_dict(image_encoder_ckpt)
            model.qformer.load_state_dict(qformer_ckpt)
            logger.info_rank0("################## Loaded DIT policy checkpoint ##################")
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
