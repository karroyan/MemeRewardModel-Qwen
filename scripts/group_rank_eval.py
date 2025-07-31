import os
import json
import yaml
import numpy as np
import torch
from itertools import combinations
from tqdm import tqdm
from typing import Optional, Any
from transformers import Seq2SeqTrainingArguments, AutoModel, AutoConfig
from llamafactory.hparams import get_infer_args, ModelArguments, DataArguments, FinetuningArguments
from llamafactory.model import load_tokenizer, AutoModelForBinaryClassification
from llamafactory.model.model_utils.classification_head import prepare_classification_model
from llamafactory.model.patcher import patch_classification_model
from llamafactory.data import get_template_and_fix_tokenizer, ClassificationDataCollatorWithPadding


def load_image_paths(dataset_path):
    """Load image paths and scores from dataset file"""
    data = json.load(open(dataset_path, 'r'))
    image_paths = []
    score = []
    for key in data['rank'].keys():
        image_paths.append(data['rank'][key])
        score.append(data['score'][key])
    return image_paths, score


def tokenize_messages(messages, images, template, tokenizer, processor):
    """Tokenize messages using the template system"""
    # For binary classification, we need both user prompt and empty assistant response
    # to match the expected format for template.encode_multiturn
    if len(messages) == 1 and messages[0]["role"] == "user":
        # Add empty assistant response for proper pairing
        messages = messages + [{"role": "assistant", "content": ""}]
    
    # Process messages through the template's multimodal plugin
    processed_messages = template.mm_plugin.process_messages(
        messages, images, [], [], processor  # images, videos, audios
    )
    
    # Get initial token IDs from multimodal plugin
    input_ids, _ = template.mm_plugin.process_token_ids(
        [], [], images, [], [], tokenizer, processor
    )
    
    # Encode the processed messages
    encoded_pairs = template.encode_multiturn(tokenizer, processed_messages, None, None)
    
    # Take the first turn (user prompt) and append to input_ids
    if encoded_pairs:
        source_ids, _ = encoded_pairs[0]
        input_ids += source_ids
    
    # Create attention mask
    attention_mask = [1] * len(input_ids)
    
    return input_ids, attention_mask


def move_to_device(batch, device):
    """Recursively move batch items to device, handling nested structures"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        for k, v in batch.items():
            batch[k] = move_to_device(v, device)
        return batch


def batch_inference(model, dataloader):
    """Run inference on batches from dataloader."""
    all_preds = []
    device = next(model.parameters()).device
    print(f"Using device for inference: {device}")
    
    for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        # Move entire batch to device, handling nested structures
        batch = move_to_device(batch, device)
        
        # Handle image_grid_thw if present
        if 'image_grid_thw' in batch:
            if isinstance(batch['image_grid_thw'], torch.Tensor):
                batch['image_grid_thw'] = batch['image_grid_thw'].to(device)
            elif isinstance(batch['image_grid_thw'], (list, tuple)):
                batch['image_grid_thw'] = [item.to(device) if isinstance(item, torch.Tensor) else item 
                                          for item in batch['image_grid_thw']]
        
        with torch.no_grad():
            outputs = model(**batch)
            # For binary classification, take argmax of logits
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
    
    return all_preds


def create_pairwise_dataset(image_paths, prompt, template, tokenizer, processor):
    """Create pairwise dataset for classification"""
    pairs = list(combinations(range(len(image_paths)), 2))
    dataset = []
    
    for idx1, idx2 in pairs:
        # Create messages in expected format
        messages = [{
            "role": "user", 
            "content": prompt
        }]
        
        # Images for this pair
        images = [image_paths[idx1], image_paths[idx2]]
        
        # Tokenize the messages
        input_ids, attention_mask = tokenize_messages(
            messages, images, template, tokenizer, processor
        )
        
        # Create properly formatted example
        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [0],  # Placeholder label for inference
            "images": images,
            "videos": [],
            "audios": []
        }
        dataset.append(example)
    
    return pairs, dataset


class PairwiseDataset(torch.utils.data.Dataset):
    """Custom dataset class for pairwise comparisons"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def build_preference_matrix(pairs, preds, n):
    preference_matrix = np.zeros((n, n), dtype=int)
    for (idx1, idx2), pred in zip(pairs, preds):
        if pred == 0:  # First image preferred
            preference_matrix[idx1, idx2] = 1
            preference_matrix[idx2, idx1] = 0
        elif pred == 1:  # Second image preferred
            preference_matrix[idx1, idx2] = 0
            preference_matrix[idx2, idx1] = 1
    return preference_matrix


def borda_count_ranking(preference_matrix):
    n = preference_matrix.shape[0]
    borda_counts = np.sum(preference_matrix, axis=1)
    ranking = np.argsort(-borda_counts)
    return ranking, borda_counts


def compute_ebc_scores(ranking):
    n = len(ranking)
    ebc_scores = np.zeros(n)
    for i in range(n):
        position = np.where(ranking == i)[0][0]
        ebc_scores[i] = 1 - (position) / n
    return ebc_scores


def kendall_tau(ranking, scores):
    from scipy.stats import kendalltau
    if scores is None:
        return None
    ranking_0_n = np.arange(len(ranking))
    tau, p_value = kendalltau(ranking, ranking_0_n)
    return tau


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='YAML config file with model paths and settings')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='group_rank_results.json')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Ensure required keys are present
    required_keys = ['model_name_or_path', 'template', 'trust_remote_code']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required keys in config: {missing_keys}")

    # Set inference-specific args
    config['stage'] = 'rm_class'  # For reward model classification
    config['finetuning_type'] = 'lora' if 'adapter_name_or_path' in config else 'full'

    # Load tokenizer and template using llamafactory
    model_args, data_args, finetuning_args, generating_args = get_infer_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    tokenizer.padding_side = "right"  # avoid overflow issue in batched inference
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Load model directly using transformers and AutoModelForBinaryClassification
    print(f"Loading base model from: {model_args.model_name_or_path}")
    
    # Determine target device
    if torch.cuda.is_available():
        target_device = "cuda:0"
    else:
        target_device = "cpu"
    print(f"Target device: {target_device}")
    
    # Load base model using transformers
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    base_model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=getattr(torch, model_args.compute_dtype) if model_args.compute_dtype else None,
        # Remove device_map to ensure model is loaded to single device
        # device_map="auto" if model_args.device_map == "auto" else None
    )
    
    # Move base model to target device
    base_model = base_model.to(target_device)
    
    # Prepare model for classification
    prepare_classification_model(base_model)
    
    # Wrap with AutoModelForBinaryClassification
    model = AutoModelForBinaryClassification.from_pretrained(base_model)
    
    # Ensure the wrapped model is also on the target device
    model = model.to(target_device)
    
    # Apply patches
    patch_classification_model(model)
    
    # Load classification head if adapter path is provided
    if model_args.adapter_name_or_path is not None:
        classification_head_path = model_args.adapter_name_or_path[-1] if isinstance(model_args.adapter_name_or_path, list) else model_args.adapter_name_or_path
    else:
        classification_head_path = model_args.model_name_or_path
    
    try:
        model.load_classification_head(classification_head_path)
        print(f"Loaded classification head from: {classification_head_path}")
        # Ensure classification head is also on the correct device
        if hasattr(model, 'classification_head'):
            model.classification_head = model.classification_head.to(target_device)
    except Exception as e:
        print(f"Warning: Could not load classification head from {classification_head_path}: {e}")
    
    model.eval()

    # Get the actual device where model parameters are located
    model_device = next(model.parameters()).device
    
    # Verify all model parameters are on the same device
    devices = set()
    for name, param in model.named_parameters():
        devices.add(param.device)
    
    if len(devices) > 1:
        print(f"Warning: Model parameters are on multiple devices: {devices}")
        print("Moving all parameters to target device...")
        model = model.to(target_device)
        # Re-check devices after moving
        devices = set()
        for name, param in model.named_parameters():
            devices.add(param.device)
        print(f"After moving, devices: {devices}")
    else:
        print(f"All model parameters are on device: {list(devices)[0]}")

    print(f"\nLoaded model:")
    print(f"Base model: {model_args.model_name_or_path}")
    if model_args.adapter_name_or_path:
        print(f"Adapter: {model_args.adapter_name_or_path}")
    print(f"Template: {data_args.template}")
    print(f"Model device: {model_device}")
    
    # Check if model parameters are on multiple devices (distributed)
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    if len(devices) > 1:
        print(f"Warning: Model is distributed across multiple devices: {devices}")
    else:
        print(f"Model is on single device: {model_device}")

    # Load data directly from JSON file
    image_paths, scores = load_image_paths(args.dataset_path)
    with open(args.prompt_path, 'r') as f:
        prompt = f.read().strip() + '\n\n\nFirst image: <image>\nSecond image:<image>'
    print(f"\nLoaded {len(image_paths)} images")

    # Create pairwise dataset with tokenized messages
    print("Tokenizing dataset...")
    pairs, dataset_data = create_pairwise_dataset(image_paths, prompt, template, tokenizer, processor)
    print(f"Created {len(pairs)} pairs")

    # Debug: Check first example
    if dataset_data:
        first_example = dataset_data[0]
        print(f"First example keys: {first_example.keys()}")
        print(f"Input IDs length: {len(first_example['input_ids'])}")
        print(f"Images: {len(first_example['images'])} images")

    # Create custom dataset
    dataset = PairwiseDataset(dataset_data)
    
    # Create data collator
    data_collator = ClassificationDataCollatorWithPadding(
        template=template,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8
    )

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    try:
        # Run inference
        preds = batch_inference(model, dataloader)

        # Build preference matrix
        preference_matrix = build_preference_matrix(pairs, preds, len(image_paths))
        
        # Compute ranking and scores
        ranking, borda_counts = borda_count_ranking(preference_matrix)
        ebc_scores = compute_ebc_scores(ranking)
        from scipy.stats import kendalltau
        
        tau = kendall_tau(ranking, scores)

        tau, p_value = kendalltau(ranking, np.arange(len(ranking)))
        print(f"p-value: {p_value:.4f}, tau: {tau:.4f}")

        # Save results
        result = {
            'image_paths': image_paths,
            'borda_counts': borda_counts.tolist(),
            'ebc_scores': ebc_scores.tolist(),
            'ranking': ranking.tolist(),
            'preference_matrix': preference_matrix.tolist(),
            'tau': tau
        }

        # Print ranking results
        print("\nImage Ranking (Best to Worst):")
        for i, idx in enumerate(ranking):
            print(f"{i+1}. {os.path.basename(image_paths[idx])} - Score: {ebc_scores[idx]:.4f}")
        print(f"\nKendall Tau: {tau:.4f}")

        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output_path}")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 