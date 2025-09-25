!

# set -x # Print bash script out as it executes

#Copy Model To This Machine (Super Fast the Second Time)
HUG_DIR=/huggingface_models
mkdir $HUG_DIR
rclone copy --progress --transfers 128  /root/CleanCode/Github/DiffSynth-Studio/huggingface_models $HUG_DIR

#Icecream equivalent for bash
# ic() { for v in "$@"; do echo "[ic] $v=${!v}"; done; }
ic(){ for v in "$@"; do echo -e "\033[1;32m[ic] $v=${!v}\033[0m"; done; }
icl(){ local name="$1"; local -n arr="$1"; echo -e "\033[1;32m[ic] $name:\033[0m"; printf "\033[1;32m  %s\033[0m\n" "${arr[@]}"; }


#Custom model path locations
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

LOW_NOISE_MODEL_PATHS='[
  [
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00002-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00003-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00004-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00005-of-00006.safetensors",
    "'"$HUG_DIR"'/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"
  ],
  "'"$HUG_DIR"'/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
  "'"$HUG_DIR"'/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
]'


export PYTHONUNBUFFERED=1 #Print Immediately


# Control debug printing ranks (all, silent, or comma-separated like 0,1,2)
DEBUG_PRINT_RANKS="all"
# DEBUG_PRINT_RANKS="silent"
# DEBUG_PRINT_RANKS="0"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0

# #Project: WEB360
# PROJECT_NAME="Web360"
# DATASET_METADATA_PATH="data/WEB360_Video_Dataset/metadata.csv"
# DATASET_BASE_PATH="data/WEB360_Video_Dataset/WEB360/videos_480x832x49"
# EXTRA_ARGS=()

#Project: GWTF-Test
PROJECT_NAME="GWTF_Dev"
DATASET_METADATA_PATH="data/envato_noisewarp_dataset/metadata.csv"
DATASET_BASE_PATH="data/envato_noisewarp_dataset/Noisewarp"
EXTRA_ARGS=(
  --use_warped_noise  # Uncomment to use pre-generated noise files instead of random noise
)
RESUME=0  # Set to 1 to resume from latest checkpoint
TRAIN_LOW_NOISE=0  # Set to 1 to train low noise model instead of high noise
ACCELERATE_ARGS=(
  #Comment out the ones you don't want to use
  --use_fsdp #Fully Sharded Data Parallel
  # --use_deepspeed #Idk what this does really...
  --mixed_precision "yes"
)

COMMON_ARGS=(
  --dataset_base_path $DATASET_BASE_PATH
  --dataset_metadata_path $DATASET_METADATA_PATH
  --height 480
  --width 832
  --num_frames 49
  --save_steps 250
  --lora_rank 512
  --dataset_repeat 100
  --learning_rate 1e-4
  --gradient_accumulation_steps 1
  --num_epochs 100
  --remove_prefix_in_ckpt pipe.dit.
  --lora_base_model dit
  --extra_inputs input_image
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --debug_print_ranks "$DEBUG_PRINT_RANKS"
)
# DEBUG_PRINT_RANKS="0"  # Only rank 0 prints
# DEBUG_PRINT_RANKS="silent"  # No debug printing
# DEBUG_PRINT_RANKS="0,1"  # Only ranks 0 and 1 print

# Set model paths and output based on noise type
if [ "$TRAIN_LOW_NOISE" = "1" ]; then
  MODEL_PATHS="$LOW_NOISE_MODEL_PATHS"
  OUTPUT_PATH="./models/train/Wan2.2-I2V-A14B_low_noise_lora$PROJECT_NAME"
  MAX_TIMESTEP=1
  MIN_TIMESTEP=0
else
  MODEL_PATHS="$HIGH_NOISE_MODEL_PATHS"
  OUTPUT_PATH="./models/train/Wan2.2-I2V-A14B_high_noise_lora$PROJECT_NAME"
  MAX_TIMESTEP=0.358
  MIN_TIMESTEP=0
fi

# Find latest checkpoint if resuming
LORA_CHECKPOINT=""
if [ "$RESUME" = "1" ]; then
  if [ -d "$OUTPUT_PATH" ]; then
    LATEST_CHECKPOINT=$(ls "$OUTPUT_PATH"/step-*.safetensors 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
      LORA_CHECKPOINT="--lora_checkpoint $LATEST_CHECKPOINT"
      ic LATEST_CHECKPOINT
    fi
  fi
fi

#Print things out
ic HUG_DIR
ic CUDA_VISIBLE_DEVICES
ic DEBUG_PRINT_RANKS
ic PROJECT_NAME
ic RESUME TRAIN_LOW_NOISE
ic MODEL_PATHS OUTPUT_PATH MAX_TIMESTEP MIN_TIMESTEP
icl HIGH_NOISE_MODEL_PATHS
icl LOW_NOISE_MODEL_PATHS
icl ACCELERATE_ARGS
icl COMMON_ARGS
icl EXTRA_ARGS

accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  examples/wanvideo/model_training/train.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  $LORA_CHECKPOINT \
  --output_path "$OUTPUT_PATH" \
  --model_paths "$MODEL_PATHS" \
  --max_timestep_boundary $MAX_TIMESTEP \
  --min_timestep_boundary $MIN_TIMESTEP

#DOCUMENTATION:
#    options:
#      -h, --help            show this help message and exit
#      --dataset_base_path DATASET_BASE_PATH
#                            Base path of the dataset.
#      --dataset_metadata_path DATASET_METADATA_PATH
#                            Path to the metadata file of the dataset.
#      --max_pixels MAX_PIXELS
#                            Maximum number of pixels per frame, used for dynamic resolution..
#      --height HEIGHT       Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.
#      --width WIDTH         Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.
#      --num_frames NUM_FRAMES
#                            Number of frames per video. Frames are sampled from the video prefix.
#      --data_file_keys DATA_FILE_KEYS
#                            Data file keys in the metadata. Comma-separated.
#      --dataset_repeat DATASET_REPEAT
#                            Number of times to repeat the dataset per epoch.
#      --model_paths MODEL_PATHS
#                            Paths to load models. In JSON format.
#      --model_id_with_origin_paths MODEL_ID_WITH_ORIGIN_PATHS
#                            Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.
#      --learning_rate LEARNING_RATE
#                            Learning rate.
#      --num_epochs NUM_EPOCHS
#                            Number of epochs.
#      --output_path OUTPUT_PATH
#                            Output save path.
#      --remove_prefix_in_ckpt REMOVE_PREFIX_IN_CKPT
#                            Remove prefix in ckpt.
#      --trainable_models TRAINABLE_MODELS
#                            Models to train, e.g., dit, vae, text_encoder.
#      --lora_base_model LORA_BASE_MODEL
#                            Which model LoRA is added to.
#      --lora_target_modules LORA_TARGET_MODULES
#                            Which layers LoRA is added to.
#      --lora_rank LORA_RANK
#                            Rank of LoRA.
#      --lora_checkpoint LORA_CHECKPOINT
#                            Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.
#      --extra_inputs EXTRA_INPUTS
#                            Additional model inputs, comma-separated.
#      --use_gradient_checkpointing_offload
#                            Whether to offload gradient checkpointing to CPU memory.
#      --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
#                            Gradient accumulation steps.
#      --max_timestep_boundary MAX_TIMESTEP_BOUNDARY
#                            Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).
#      --min_timestep_boundary MIN_TIMESTEP_BOUNDARY
#                            Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).
#      --find_unused_parameters
#                            Whether to find unused parameters in DDP.
#      --save_steps SAVE_STEPS
#                            Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.
#      --dataset_num_workers DATASET_NUM_WORKERS
#                            Number of workers for data loading.
#      --weight_decay WEIGHT_DECAY
#                            Weight decay.
#      --debug_print_line_tracing
#                            Enable ultra-verbose line tracing (prints every executed line).
#      --debug_print_ranks DEBUG_PRINT_RANKS
#                            Which ranks to print debug messages from. Options: 'all', 'silent' (alias for ''),
#                            comma-separated rank numbers like '0,1,2' (default: 'all').
#      --use_warped_noise    Use pre-generated noise files instead of random noise generation.
#                            Requires 'noise' field in dataset pointing to .npy files.
#
# WARPED NOISE DATASET CREATION:
# To create a warped noise dataset like the Envato dataset:
# 1. Install dependencies: pip install rp
# 2. Use rp.git.CommonSource.noise_warp.get_noise_from_video() to extract noise from videos
# 3. Expected format: .npy files with shape (81, 60, 104, 16) = (T, H, W, C)
# 4. CSV metadata should include 'noise' column pointing to the .npy files
# Example CSV:
#   video,prompt,noise
#   video001.mp4,"A beautiful sunset",noise001.npy
#   video002.mp4,"Ocean waves",noise002.npy
# The noise files are automatically resized to match training parameters (T, H, W, C) -> (B, C, T', H', W')
