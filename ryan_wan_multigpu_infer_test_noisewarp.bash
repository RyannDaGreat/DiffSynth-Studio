!

# Copy Model To This Machine (Super Fast the Second Time)
HUG_DIR=/huggingface_models
mkdir -p $HUG_DIR
rclone copy --progress --transfers 128 /root/CleanCode/Github/DiffSynth-Studio/huggingface_models $HUG_DIR

# Icecream equivalent for bash
ic() { for v in "$@"; do echo -e "\033[1;32m[ic] $v=${!v}\033[0m"; done; }

# Define LoRA checkpoints from rp call download_to_cache
LORA_DIT=$( rp call download_to_cache --- "models/train/Wan2.2-I2V-A14B_high_noise_loraGWTF_Dev/step-2750.safetensors" --show_progress True)
LORA_DIT2=$(rp call download_to_cache --- "models/train/Wan2.2-I2V-A14B_low_noise_lora_GWTF_Dev/step-2750.safetensors" --show_progress True)

# Choose the content
PROMPT="A graceful tabby cat with distinctive striped markings carefully climbs down from a tall tree, moving with feline agility and precision. The cat grips the rough bark with its claws, methodically placing each paw as it descends through the dappled sunlight filtering through green leaves. Its alert eyes scan the ground below while its fluffy tail sways for balance in this natural outdoor woodland setting"
OUTPUT="cat_climbing_down_tree.mp4"
INPUT_IMAGE_PATH="/root/CleanCode/Sandbox/wan_gwtf_test/cat_off_tree_input_video_480x832.png"

# Custom noise file for warped noise
WARPED_NOISE="/root/CleanCode/Sandbox/wan_gwtf_test/cat_off_tree_input_video_480x832/noises.npy"  # Shape: (49, 60, 104, 16) = (T, H, W, C)
DEGRADATION_ALPHA=0.0  # 0.0 = pure custom noise, 1.0 = pure random, unset = random alpha

OUTPUT=$(rp call get_unique_copy_path --- "$OUTPUT")
INPUT_IMAGE_PATH=$(rp call download_to_cache --- "$INPUT_IMAGE_PATH")

ic LORA_DIT LORA_DIT2 HUG_DIR PROMPT OUTPUT INPUT_IMAGE_PATH WARPED_NOISE DEGRADATION_ALPHA

# Run inference with custom noise
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes 8 --multi_gpu ryan_wan_multigpu_infer_test.py \
    --root "$HUG_DIR/Wan2.2-I2V-A14B" \
    --lora_dit "$LORA_DIT" \
    --lora_dit2 "$LORA_DIT2" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --input_image_path "$INPUT_IMAGE_PATH" \
    --warped_noise "$WARPED_NOISE" \
    --degradation_alpha "$DEGRADATION_ALPHA"

rp call fansi_print --- "OUTPUT = $OUTPUT" "green green bold italic on dark dark blue"

