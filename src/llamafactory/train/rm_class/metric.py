# Copyright 2025 the LlamaFactory team.
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
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r"""Compute reward accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        chosen_scores, rejected_scores = numpify(eval_preds.predictions[0]), numpify(eval_preds.predictions[1])
        if not chosen_scores.shape:
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)
        else:
            for i in range(len(chosen_scores)):
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])

        if compute_result:
            return self._dump()

@dataclass
class ComputeClassificationAccuracy:
    r"""Compute binary classification accuracy."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {"accuracy": float(np.mean(self.score_dict["accuracy"]))}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        """
        Compute binary classification accuracy.
        
        Args:
            eval_preds: EvalPrediction object containing predictions and labels
        """
        # Extract predictions and labels
        predictions = eval_preds.predictions
        true_labels = eval_preds.label_ids
        
        # Handle different prediction formats
        if isinstance(predictions, tuple) and len(predictions) == 2:
            # Format: (logits, labels) from prediction_step
            logits, pred_labels = predictions
            logits = numpify(logits)
            true_labels = numpify(pred_labels) if pred_labels is not None else numpify(true_labels)
        else:
            # Direct logits format
            logits = numpify(predictions)
            true_labels = numpify(true_labels)
        
        # Convert logits to predictions
        if logits.ndim == 2 and logits.shape[1] == 2:
            # Binary classification logits [batch_size, 2]
            predicted_labels = np.argmax(logits, axis=1)
        elif logits.ndim == 1:
            # Single logit per sample - threshold at 0
            predicted_labels = (logits > 0).astype(int)
        else:
            # Handle score format (chosen_scores, rejected_scores) - for compatibility
            if isinstance(eval_preds.predictions, tuple) and len(eval_preds.predictions) == 2:
                chosen_scores, rejected_scores = eval_preds.predictions
                chosen_scores = numpify(chosen_scores)
                rejected_scores = numpify(rejected_scores)
                predicted_labels = (chosen_scores > rejected_scores).astype(int)
                true_labels = numpify(true_labels)
            else:
                raise ValueError(f"Unexpected predictions format: {type(predictions)}")
        
        # Handle true labels format
        if true_labels.ndim == 2:
            true_labels = true_labels[:, 0]  # Take first column if 2D
        
        # Ensure labels are integers
        true_labels = true_labels.astype(int)
        predicted_labels = predicted_labels.astype(int)
        
        # Compute accuracy
        correct_predictions = (predicted_labels == true_labels)
        accuracy = float(np.mean(correct_predictions))
        
        # Store results for batch processing
        for correct in correct_predictions:
            self.score_dict["accuracy"].append(correct)

        if compute_result:
            return {"accuracy": accuracy}