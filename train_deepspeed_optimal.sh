#!/bin/bash

# Optimal DeepSpeed training script for 8x A100 80GB with 1.1TB RAM

# set -x # Uncomment for debugging

# Setup environment
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATE_USE_DEEPSPEED=true  # Enable DeepSpeed detection in our code

# Optional: Use ZeRO-3 for maximum memory efficiency (slower but uses less memory)
# export USE_ZERO3=true

echo "=========================================="
echo "DeepSpeed Training Configuration"
echo "=========================================="
echo "GPUs: 8x A100 80GB"
echo "RAM: 1.1TB"
echo "CPUs: 96 cores"
echo "DeepSpeed: ZeRO-2 (optimal balance)"
echo "=========================================="

# Copy models if needed
HUG_DIR=/huggingface_models
mkdir -p $HUG_DIR

# Model paths
HIGH_NOISE_MODEL_PATHS='[
  [
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"
  ],
  "'"$HUG_DIR"'/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
  "'"$HUG_DIR"'/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
]'

# Training configuration
PROJECT_NAME="GWTF_Dev_DeepSpeed_Optimal"
DATASET_METADATA_PATH="data/envato_noisewarp_dataset/metadata.csv"
DATASET_BASE_PATH="data/envato_noisewarp_dataset/Noisewarp"
OUTPUT_PATH="./models/train/Wan2.2-I2V-A14B_high_noise_lora_$PROJECT_NAME"

# Launch training with DeepSpeed
accelerate launch \
  --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --deepspeed_config_file deepspeed_config_optimal.json \
  --zero_stage 2 \
  --gradient_accumulation_steps 1 \
  --gradient_clipping 1.0 \
  --offload_optimizer_device cpu \
  --offload_param_device cpu \
  --deepspeed_multinode_launcher standard \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path $DATASET_BASE_PATH \
  --dataset_metadata_path $DATASET_METADATA_PATH \
  --height 480 \
  --width 832 \
  --num_frames 81 \
  --save_steps 250 \
  --lora_rank 512 \
  --dataset_repeat 100 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 1 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt pipe.dit. \
  --lora_base_model dit \
  --extra_inputs input_image \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --output_path "$OUTPUT_PATH" \
  --model_paths "$HIGH_NOISE_MODEL_PATHS" \
  --max_timestep_boundary 0.358 \
  --min_timestep_boundary 0 \
  --use_warped_noise \
  --debug_print_ranks "0"

echo "Training complete!"
