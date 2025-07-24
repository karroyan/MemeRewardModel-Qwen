#!/bin/bash

# Base model and adapter paths
BASE_MODEL_PATH="/fs-computility/niuyazhe/lixueyan/meme/LLaMA-Factory/output/binary_classification_merged"
# ADAPTER_PATH="/fs-computility/niuyazhe/lixueyan/meme/LLaMA-Factory/output/binary_classification_merged"

# Dataset and prompt paths
DATASET_PATH="/fs-computility/niuyazhe/shared/meme/data/meme/rmdata_forhuman/d4/human_rank.json"
PROMPT_PATH="/fs-computility/niuyazhe/lixueyan/meme/dataset-meme-rewardmodel/prompt/reward_model_prompt.txt"

# Output path
OUTPUT_PATH="ebc_ranking_results_singlehead_sample_0411_1000.json"

# Model config
cat > temp_config.yaml << EOL
model_name_or_path: ${BASE_MODEL_PATH}
adapter_name_or_path: ${ADAPTER_PATH}
template: qwen2_vl
trust_remote_code: true
image_max_pixels: 262144
video_max_pixels: 16384
infer_backend: huggingface
EOL

# Run the evaluation
python scripts/group_rank_eval.py \
    --config temp_config.yaml \
    --dataset_path ${DATASET_PATH} \
    --prompt_path ${PROMPT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --batch_size 8

# Clean up
rm temp_config.yaml 