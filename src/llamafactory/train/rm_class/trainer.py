# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import Trainer
from typing_extensions import override

from ...model import load_config, load_model, load_tokenizer, AutoModelForBinaryClassification
from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class PairwiseTrainer(Trainer):
    r"""Inherits Trainer to compute pairwise loss."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

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
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Compute cross entropy loss for binary classification. The model takes one input with two images and outputs classification logits.

        Subclass and override to inject custom behavior.

        Note that the first element will be removed from the output tuple.
        See: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py#L3842
        """
        # Model forward pass - expecting classification logits as output
        outputs = model(**inputs, return_dict=True)
        
        # Extract logits from model output
        # Expecting the model to output logits of shape [batch_size, 2] for binary classification
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  # Shape: [batch_size, 2]
        else:
            raise ValueError(f"Model output does not contain 'logits' attribute: {outputs}")
        
        # Get labels for cross entropy loss
        if "labels" in inputs:
            labels = inputs["labels"]  # Shape: [batch_size], values should be 0 or 1
        else:
            raise ValueError("Labels are required for cross entropy loss but not found in inputs")
        
        # Compute cross entropy loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels[:, 0])
        
        if return_outputs:
            # For evaluation, return logits and labels
            if not self.model.training:
                return loss, (logits, labels)
            else:
                # For training, return compatibility format (chosen_scores, rejected_scores)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                chosen_scores = probabilities[:, 1]  # Probability of class 1 (preferred)
                rejected_scores = probabilities[:, 0]  # Probability of class 0 (not preferred)
                return loss, (chosen_scores, rejected_scores)
        else:
            return loss

    def prediction_step(
        self,
        model: "PreTrainedModel",
        inputs: dict[str, "torch.Tensor"],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Custom prediction step to return logits for evaluation.
        """
        has_labels = "labels" in inputs
        
        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits
            
            if has_labels:
                labels = inputs["labels"]
                if labels.ndim == 2:
                    labels = labels[:, 0]
                
                # Compute loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            else:
                loss = None
                labels = None
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, labels)
    
    # Note: Removed save_predictions method as it's not needed for simple accuracy evaluation
