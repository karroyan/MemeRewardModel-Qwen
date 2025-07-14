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

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class BinaryClassificationDatasetProcessor(DatasetProcessor):
    """
    Processor for binary classification tasks with two images.
    Expected data format:
    {
        "messages": [
            {
                "role": "user",
                "content": "Compare these two images"
            },
            {
                "role": "assistant", 
                "content": "Classification response"  # Optional, can be empty
            }
        ],
        "images": ["image1.jpg", "image2.jpg"],  # Exactly 2 images
        "labels": 0  # Binary label: 0 or 1
    }
    """
    
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        label: int,
    ) -> tuple[list[int], list[int], int]:
        """
        Encode data example for binary classification.
        Returns: (input_ids, attention_mask, label)
        """
        # Process the prompt through the template
        messages = self.template.mm_plugin.process_messages(prompt+response, images, videos, audios, self.processor)
        input_ids, _ = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        
        # For classification, we only need the prompt part
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        
        # Take only the first turn (user prompt)
        if encoded_pairs:
            source_ids, _ = encoded_pairs[0]
            # Truncate if too long
            if len(source_ids) > self.data_args.cutoff_len:
                source_ids = source_ids[:self.data_args.cutoff_len]
            
            input_ids += source_ids
        
        # For classification, we don't need response tokens, just the input
        attention_mask = [1] * len(input_ids)
        
        return input_ids, attention_mask, label

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        Preprocess dataset for binary classification.
        """
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_prompt"])):
            # Validate data format
            if not examples["_images"][i] or len(examples["_images"][i]) != 2:
                logger.warning_rank0(
                    f"Dropped invalid example: expected exactly 2 images, got {len(examples['_images'][i] or [])}"
                )
                continue
                
            if "_labels" not in examples or examples["_labels"][i] is None:
                logger.warning_rank0("Dropped example without classification label")
                continue
                
            label = examples["_labels"][i]
            if label[0] not in [0, 1]:
                logger.warning_rank0(f"Dropped example with invalid label: {label} (expected 0 or 1)")
                continue

            # try:
            input_ids, attention_mask, classification_label = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i] if examples["_response"][i] else [],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                label=examples["_labels"][i],
            )
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(classification_label)  # Binary classification label
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
                
            # except Exception as e:
            #     logger.warning_rank0(f"Failed to process example {i}: {e}")
            #     continue

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print a data example for debugging."""
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("classification_label:\n{}".format(example["labels"]))
        print("images:\n{}".format(example.get("images", []))) 