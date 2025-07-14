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

from typing import TYPE_CHECKING, Any, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import cached_file

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


class AttentionPooling(nn.Module):
    """
    Overview:
        Attention pooling layer on the sequence dimension of LLM/VLM hidden states.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        position_bias: bool = False,
        position_bias_scale: float = 3.0,
    ):
        super(AttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.position_bias = position_bias
        self.position_bias_scale = position_bias_scale

        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        # 0.02 for better initialization
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden_states):
        B, S, C = hidden_states.shape

        # Multi-head projection for key and value
        k = self.k(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D
        v = self.v(hidden_states).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B, H, S, D

        # Expand query for batch dimension
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # B, H, C
        q = q.unsqueeze(2)  # B, H, 1, C
        q = q.reshape(B, self.num_heads, 1, self.head_dim)  # B, H, 1, C

        # Attention weights
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, H, 1, S

        # Add position bias
        if self.position_bias:
            position_bias = torch.arange(S, device=k.device).float() / S * self.position_bias_scale
            attn = attn + position_bias.view(1, 1, 1, -1)  # Add position bias

        # Attention pooling
        attn = torch.softmax(attn, dim=-1)  # B, H, 1, S
        out = (attn @ v).squeeze(2)  # B, H, D
        out = out.reshape(B, -1)  # B, C

        return out

@dataclass
class ClassificationOutput(ModelOutput):
    """
    Output class for binary classification models.
    
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Classification logits (before softmax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the base model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attentions from the base model.
    """
    logits: torch.FloatTensor = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


class AutoModelForBinaryClassification(PreTrainedModel):
    """
    Model wrapper that adds a binary classification head to any pretrained model.
    """
    
    def __init__(self, pretrained_model: PreTrainedModel):
        super().__init__(pretrained_model.config)
        self.pretrained_model = pretrained_model
        
        # Get the hidden size from the model config
        if hasattr(pretrained_model.config, 'hidden_size'):
            hidden_size = pretrained_model.config.hidden_size
        elif hasattr(pretrained_model.config, 'd_model'):
            hidden_size = pretrained_model.config.d_model
        elif hasattr(pretrained_model.config, 'n_embd'):
            hidden_size = pretrained_model.config.n_embd
        else:
            raise ValueError("Cannot determine hidden size from model config")
        
        # Attention pooling layer
        self.attention_pooling = AttentionPooling(
            hidden_size=hidden_size,
            num_heads=4,
            qkv_bias=False,
            position_bias=True,
            position_bias_scale=3.0,
        )
        
        # Binary classification head
        self.classification_head = nn.Linear(hidden_size, 2)
        
        # Initialize the classification head
        nn.init.normal_(self.classification_head.weight, std=0.02)
        nn.init.zeros_(self.classification_head.bias)

    @classmethod
    def from_pretrained(cls, pretrained_model: PreTrainedModel) -> "AutoModelForBinaryClassification":
        """Create a binary classification model from a pretrained model."""
        return cls(pretrained_model)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, ClassificationOutput]:
        """
        Forward pass through the model with binary classification head.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through the base model
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=True,
            **kwargs,
        )
        
        # Extract the last hidden state
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            raise ValueError("Cannot extract hidden states from model output")
        
        # Use attention pooling to aggregate hidden states
        pooled_output = self.attention_pooling(hidden_states)
        
        # Pass through classification head
        logits = self.classification_head(pooled_output)
        
        if not return_dict:
            output = (logits,)
            if output_hidden_states:
                output = output + (outputs.hidden_states,)
            if output_attentions:
                output = output + (outputs.attentions,)
            return output

        return ClassificationOutput(
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the model and classification head."""
        # Save the base model
        self.pretrained_model.save_pretrained(save_directory, **kwargs)
        
        # Save the classification head and attention pooling
        classification_head_state = {
            'classification_head': self.classification_head.state_dict(),
            'attention_pooling': self.attention_pooling.state_dict(),
        }
        classification_head_path = f"{save_directory}/classification_head.pt"
        torch.save(classification_head_state, classification_head_path)
        logger.info_rank0(f"Saved classification head and attention pooling to {classification_head_path}")

    def load_classification_head(self, model_path: str):
        """Load classification head parameters."""
        classification_head_path = f"{model_path}/classification_head.pt"
        try:
            state_dict = torch.load(classification_head_path, map_location="cpu")
            
            # Handle both old format (just classification head) and new format (with attention pooling)
            if isinstance(state_dict, dict) and 'classification_head' in state_dict:
                # New format with attention pooling
                self.classification_head.load_state_dict(state_dict['classification_head'])
                if 'attention_pooling' in state_dict:
                    self.attention_pooling.load_state_dict(state_dict['attention_pooling'])
                logger.info_rank0(f"Loaded classification head and attention pooling from {classification_head_path}")
            else:
                # Old format (just classification head weights)
                self.classification_head.load_state_dict(state_dict)
                logger.info_rank0(f"Loaded classification head from {classification_head_path} (attention pooling initialized randomly)")
        except Exception as e:
            logger.warning_rank0(f"Failed to load classification head: {e}")


def prepare_classification_model(model: "PreTrainedModel") -> None:
    """Prepare model for binary classification by adding necessary attributes."""
    # Add any model-specific preparation here
    if getattr(model.config, "model_type", None) == "llava":
        setattr(model, "lm_head", model.language_model.get_output_embeddings())
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if getattr(model.config, "model_type", None) == "chatglm":
        setattr(model, "lm_head", model.transformer.output_layer)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

    if getattr(model.config, "model_type", None) == "internlm2":
        setattr(model, "lm_head", model.output)
        setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"]) 