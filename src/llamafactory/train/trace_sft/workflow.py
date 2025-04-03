# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Dict, Any

import torch
from functools import partial

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor, ComputeEgoPlanAccuracy
from .trainer import CustomSeq2SeqTrainer
from ..callbacks import SaveLastCheckpointCallback
from .trace_model import Base_TraceConfig

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


@dataclass
class SFTDataCollatorWithTrace(SFTDataCollatorWith4DAttentionMask):
    r"""
    Data collator for supporting data_args.dataset_kept_columns
    """
    kept_columns: List[str] = None
    num_trace_points: int = 16

    def __post_init__(self):
        trace_tokens = ''.join([f'<trace{i}>' for i in range(self.num_trace_points)])
        self.trace_token_ids = torch.tensor(self.tokenizer.encode(trace_tokens))

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        kept_features = {}
        for kept_col in self.kept_columns:
            kept_features[kept_col] = [feature.pop(kept_col) for feature in features]

        features = super().__call__(features)
        features.update(kept_features)

        # custom handling for future_trace
        trace_labels = []
        trace_masks = []
        trace_token_indices = []
        future_trace = features.pop('future_trace')
        for idx, trace in enumerate(future_trace):
            if trace is None:
                trace_labels.append(torch.zeros((self.num_trace_points, 2)))
                trace_masks.append(1)
                trace_token_indices.append(torch.zeros(self.num_trace_points))
            else:
                trace_labels.append(torch.tensor(trace))
                trace_masks.append(0)
                # find the trace token indices for each sample
                trace_token_indices.append(self._find_trace_token_indices(features['input_ids'][idx]))
        # add to features
        features['trace_labels'] = torch.stack(trace_labels)
        features['trace_masks'] = torch.tensor(trace_masks).to(torch.long)
        features['trace_token_indices'] = torch.stack(trace_token_indices).to(torch.long)

        return features

    def _find_trace_token_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        trace_len = self.trace_token_ids.size(0)
        candidate = (input_ids == self.trace_token_ids[0]).nonzero(as_tuple=False)
        if candidate.numel() != 1:
            raise ValueError(f"Trace token {self.trace_token_ids[0]} not found or multiple found in input_ids: {input_ids}")
        start_index = candidate[0].item()
        segment = input_ids[start_index:start_index + trace_len]
        if not torch.equal(segment, self.trace_token_ids):
            raise ValueError(f"Trace token sequence does not match starting at index {start_index}. Expected: {self.trace_token_ids}, Got: {segment}")

        return torch.arange(start_index, start_index+trace_len)


def freeze_tokens_gradients(grad, freeze_mask):
    grad[freeze_mask.to(grad.device)] = 0.


def run_trace_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    is_trace_model = issubclass(model.config_class, Base_TraceConfig)

    # freeze the embedding of the original tokens
    if is_trace_model:
        input_embeddings = model.get_input_embeddings()
        if hasattr(input_embeddings, "num_embeddings"):
            model_embedding_size = input_embeddings.num_embeddings
        else:
            model_embedding_size = input_embeddings.weight.shape[0]
        freeze_mask = torch.ones(model_embedding_size, dtype=torch.bool)
        for idx in range(model.num_trace_points):
            freeze_mask[tokenizer.encode(f'<trace{idx}>')[0]] = False
        model.get_input_embeddings().weight.register_hook(partial(freeze_tokens_gradients, freeze_mask=freeze_mask))

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    if is_trace_model:
        data_collator = SFTDataCollatorWithTrace(
            template=template,
            model=model if not training_args.predict_with_generate else None,
            pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            block_diag_attn=model_args.block_diag_attn,
            attn_implementation=getattr(model.config, "_attn_implementation", None),
            compute_dtype=model_args.compute_dtype,
            **tokenizer_module,
            kept_columns=data_args.dataset_kept_columns,
        )
    else:
        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=template,
            model=model if not training_args.predict_with_generate else None,
            pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            block_diag_attn=model_args.block_diag_attn,
            attn_implementation=getattr(model.config, "_attn_implementation", None),
            compute_dtype=model_args.compute_dtype,
            **tokenizer_module,
        )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        if 'egoplan' in data_args.eval_dataset[0]:
            metric_module["compute_metrics"] = ComputeEgoPlanAccuracy(tokenizer=tokenizer)
        else:
            metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    if finetuning_args.save_last_ckpt_steps > 0:
        trainer.add_callback(SaveLastCheckpointCallback(trainer, finetuning_args.save_last_ckpt_steps))

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
