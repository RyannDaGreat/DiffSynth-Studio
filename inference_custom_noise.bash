!

# Copy Model To This Machine (Super Fast the Second Time)
HUG_DIR=/huggingface_models
mkdir -p $HUG_DIR
rclone copy --progress --transfers 128 /root/CleanCode/Github/DiffSynth-Studio/huggingface_models $HUG_DIR

# Icecream equivalent for bash
ic() { for v in "$@"; do echo -e "\033[1;32m[ic] $v=${!v}\033[0m"; done; }

# Define LoRA checkpoints from rp call download_to_cache
LORA_DIT=$( rp call download_to_cache --- "models/train/Wan2.2-I2V-A14B_high_noise_lora_WEB360/step-8700.safetensors" --show_progress True)
LORA_DIT2=$(rp call download_to_cache --- "models/train/Wan2.2-I2V-A14B_low_noise_lora_WEB360/step-4000.safetensors"  --show_progress True)

# Choose the content
PROMPT="A ultra-realistic video of a serene mountain lake at sunrise, with mist rising from the water and golden light reflecting on the surface"
OUTPUT="mountain_lake_sunrise.mp4"
INPUT_IMAGE_PATH="https://images.unsplash.com/photo-1506905925346-21bda4d32df4"

# Custom noise file (optional)
NOISE_FILE="custom_noise.npy"  # Leave empty to use random noise
DEGRADATION=0.0  # 0.0 = pure custom noise, 1.0 = pure random

OUTPUT=$(rp call get_unique_copy_path --- "$OUTPUT")
INPUT_IMAGE_PATH=$(rp call download_to_cache --- "$INPUT_IMAGE_PATH")

ic LORA_DIT LORA_DIT2 HUG_DIR PROMPT OUTPUT INPUT_IMAGE_PATH NOISE_FILE DEGRADATION

# Run inference with custom noise
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu ryan_wan_multigpu_infer_test.py \
    --root "$HUG_DIR/Wan2.2-I2V-A14B" \
    --lora_dit "$LORA_DIT" \
    --lora_dit2 "$LORA_DIT2" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --input_image_path "$INPUT_IMAGE_PATH" \
    --noise_file "$NOISE_FILE" \
    --degradation "$DEGRADATION"