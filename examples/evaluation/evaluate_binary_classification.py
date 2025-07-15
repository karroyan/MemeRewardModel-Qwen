#!/usr/bin/env python3
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

"""
Evaluation script for binary classification models.
Evaluates trained models on multiple datasets and computes accuracy, precision, recall, and F1.
"""

import argparse
import json
import os
from typing import Dict, List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoProcessor
from transformers.trainer_utils import EvalPrediction

# Add LlamaFactory to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, ClassificationDataCollatorWithPadding
from llamafactory.model import load_model, load_tokenizer, AutoModelForBinaryClassification
from llamafactory.hparams import ModelArguments, DataArguments, FinetuningArguments
from llamafactory.train.rm_class.metric import ComputeClassificationAccuracy
from llamafactory.train.rm_class.trainer import PairwiseTrainer
from transformers import Seq2SeqTrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary classification model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the trained model (checkpoint or merged model)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        help="Path to base model (if evaluating checkpoint)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen2_vl",
        help="Template name"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["boring_detailed_train", "boringmeme_train", "irrelevantmeme_train", 
                "lowperformancememe_train", "object_add_train", "object_changed_train", "text_replaced_train"],
        help="List of datasets to evaluate on"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum number of samples to evaluate (for testing)"
    )
    
    return parser.parse_args()


def evaluate_dataset(
    model: AutoModelForBinaryClassification,
    tokenizer_module: dict,
    template,
    dataset_name: str,
    data_args: DataArguments,
    training_args: Seq2SeqTrainingArguments,
    finetuning_args: FinetuningArguments,
    model_args: ModelArguments
) -> Dict[str, float]:
    """Evaluate model on a single dataset."""
    
    print(f"\n=== Evaluating on {dataset_name} ===")
    
    # Update data_args for this dataset
    data_args.dataset = [dataset_name]
    data_args.eval_dataset = None
    
    # Load dataset
    try:
        dataset_module = get_dataset(
            template, model_args, data_args, training_args, 
            stage="rm_class", **tokenizer_module
        )
        
        if dataset_module["train_dataset"] is None:
            print(f"Warning: No data found for dataset {dataset_name}")
            return {}
            
        eval_dataset = dataset_module["train_dataset"]  # Use train split for evaluation
        print(f"Loaded {len(eval_dataset)} samples from {dataset_name}")
        
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return {}
    
    # Create data collator
    data_collator = ClassificationDataCollatorWithPadding(
        template=template, model=model, pad_to_multiple_of=8, **tokenizer_module
    )
    
    # Create trainer for evaluation
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=ComputeClassificationAccuracy(),
        finetuning_args=finetuning_args,
        processor=tokenizer_module.get("processor"),
    )
    
    # Run evaluation
    try:
        eval_results = trainer.evaluate()
        
        # Extract metrics
        metrics = {
            "accuracy": eval_results.get("eval_accuracy", 0.0),
            "precision": eval_results.get("eval_precision", 0.0), 
            "recall": eval_results.get("eval_recall", 0.0),
            "f1": eval_results.get("eval_f1", 0.0),
            "loss": eval_results.get("eval_loss", 0.0),
            "num_samples": len(eval_dataset)
        }
        
        print(f"Results for {dataset_name}:")
        for metric, value in metrics.items():
            if metric != "num_samples":
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
                
        return metrics
        
    except Exception as e:
        print(f"Error evaluating dataset {dataset_name}: {e}")
        return {}


def main():
    args = parse_args()
    
    print("=== Binary Classification Model Evaluation ===")
    print(f"Model path: {args.model_path}")
    print(f"Datasets: {args.datasets}")
    
    # Create arguments
    model_args = ModelArguments(
        model_name_or_path=args.base_model_path or args.model_path,
        adapter_name_or_path=[args.model_path] if args.base_model_path else None,
        trust_remote_code=True,
    )
    
    data_args = DataArguments(
        template=args.template,
        dataset_dir=args.dataset_dir,
        max_samples=args.max_samples,
    )
    
    finetuning_args = FinetuningArguments(
        stage="rm_class",
        finetuning_type="lora" if args.base_model_path else "full",
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        do_eval=True,
    )
    
    # Load tokenizer and model
    print("\n=== Loading Model ===")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    model = load_model(
        tokenizer, model_args, finetuning_args, 
        is_trainable=False, add_classification_head=True
    )
    model.eval()
    
    print(f"Model loaded successfully: {type(model).__name__}")
    
    # Evaluate on each dataset
    all_results = {}
    total_samples = 0
    total_correct = 0
    
    for dataset_name in args.datasets:
        results = evaluate_dataset(
            model, tokenizer_module, template, dataset_name,
            data_args, training_args, finetuning_args, model_args
        )
        
        if results:
            all_results[dataset_name] = results
            
            # Update overall stats
            total_samples += results["num_samples"]
            total_correct += results["accuracy"] * results["num_samples"]
    
    # Compute overall metrics
    if total_samples > 0:
        overall_accuracy = total_correct / total_samples
        all_results["overall"] = {
            "accuracy": overall_accuracy,
            "total_samples": total_samples,
            "datasets_evaluated": len([k for k in all_results.keys() if k != "overall"])
        }
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for dataset_name, results in all_results.items():
        if dataset_name == "overall":
            continue
        print(f"\n{dataset_name}:")
        print(f"  Samples: {results['num_samples']}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
    
    if "overall" in all_results:
        print(f"\nOVERALL:")
        print(f"  Total Samples: {all_results['overall']['total_samples']}")
        print(f"  Overall Accuracy: {all_results['overall']['accuracy']:.4f}")
        print(f"  Datasets Evaluated: {all_results['overall']['datasets_evaluated']}")
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main() 