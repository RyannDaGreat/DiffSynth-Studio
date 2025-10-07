!

# Copy Model To This Machine (Super Fast the Second Time)
HUG_DIR=/huggingface_models
mkdir -p $HUG_DIR
rclone copy --progress --transfers 128 /root/CleanCode/Github/DiffSynth-Studio/huggingface_models $HUG_DIR

# Icecream equivalent for bash
ic() { for v in "$@"; do echo -e "\033[1;32m[ic] $v=${!v}\033[0m"; done; }

# Video parameters
NUM_FRAMES=49
HEIGHT=480
WIDTH=832
CFG_SCALE=5
NUM_INFERENCE_STEPS=10
SEED=41

# Define LoRA checkpoints from rp call download_to_cache
#STEP=2000
#STEP=1250
#STEP=500
#CKPT_NAME="Debug2" ; STEP=2000
CKPT_NAME="<LR=1e-5>" ; STEP=6750
CKPT_NAME="Deepspeed_<T=81>" ; STEP=1750 ; NUM_FRAMES=81
LORA_DIT_PATH="models/train/Wan2.2-I2V-A14B_high_noise_lora_GWTF_Dev_$CKPT_NAME/step-$STEP.safetensors"
LORA_DIT2_PATH="models/train/Wan2.2-I2V-A14B_low_noise_lora_GWTF_Dev_$CKPT_NAME/step-$STEP.safetensors"
LORA_DIT=$( rp call download_to_cache --- "$LORA_DIT_PATH" --show_progress True)
LORA_DIT2=$(rp call download_to_cache --- "$LORA_DIT2_PATH" --show_progress True)

# Extract checkpoint numbers from both LoRA paths
CHECKPOINT_HIGH=$(echo "$LORA_DIT_PATH" | grep -o 'step-[0-9]*' | sed 's/step-//')
CHECKPOINT_LOW=$(echo "$LORA_DIT2_PATH" | grep -o 'step-[0-9]*' | sed 's/step-//')

# Choose the content

#CAT CLIMBS DOWN TREE
PROMPT="A graceful tabby cat with distinctive striped markings carefully climbs down from a tall tree, moving with feline agility and precision. The cat grips the rough bark with its claws, methodically placing each paw as it descends through the dappled sunlight filtering through green leaves. Its alert eyes scan the ground below while its fluffy tail sways for balance in this natural outdoor woodland setting"
NAME="cat_off_tree_input_video_480x832"

# #DOG WATCHES BUTTERFLY
# PROMPT="A puppy looks at a butterfly curiously, its head moving from left to right as the butterfly flutters in the wind"
# NAME='doggy_butterfly'

#CORGI BARKS
PROMPT="A happy Corgi sits indoors, looking directly at the camera with bright, eager eyes. It opens and closes its mouth twice, letting out two  slow-motion cheerful barks. The dog’s expression stays joyful, radiating playful energy"
NAME="corgi_cutanddrag_81f_480p"

DEGRADATION_ALPHA=0  # 0 = pure custom noise, 1 = pure random, unset = random alpha
DEGRADATION_ALPHA=.01  # 0 = pure custom noise, 1 = pure random, unset = random alpha
DEGRADATION_ALPHA=.3  # 0 = pure custom noise, 1 = pure random, unset = random alpha
#DEGRADATION_ALPHA=.4  # 0 = pure custom noise, 1 = pure random, unset = random alpha
#DEGRADATION_ALPHA=.5  # 0 = pure custom noise, 1 = pure random, unset = random alpha
#DEGRADATION_ALPHA=.6  # 0 = pure custom noise, 1 = pure random, unset = random alpha
#DEGRADATION_ALPHA=.75  # 0 = pure custom noise, 1 = pure random, unset = random alpha
#DEGRADATION_ALPHA=1  # 0 = pure custom noise, 1 = pure random, unset = random alpha

#Calculate warped noise if we haven't already
NOISEWARPS_ROOT="test_noise_warpings"
INPUT_VIDEO_PATH="$NOISEWARPS_ROOT/$NAME.mp4"
NOISEWARP_FOLDER="${NOISEWARPS_ROOT}/${NAME}_480x832"
INPUT_IMAGE_PATH="$NOISEWARP_FOLDER/first_frame.png"  
WARPED_NOISE="$NOISEWARP_FOLDER/noises.npy"  
python test_noise_warpings/noise_warper.py "$INPUT_VIDEO_PATH" "$NOISEWARP_FOLDER"

# Generate output filename with parameters
OUTPUT="${NAME}_<${HEIGHT}×${WIDTH}×${NUM_FRAMES},CFG=${CFG_SCALE},N=${NUM_INFERENCE_STEPS},S=${SEED},D=${DEGRADATION_ALPHA},HI=${CHECKPOINT_HIGH},LO=${CHECKPOINT_LOW},CP=${CKPT_NAME}>.mp4"
OUTPUT=$(rp call get_unique_copy_path --- "$OUTPUT")
INPUT_IMAGE_PATH=$(rp call download_to_cache --- "$INPUT_IMAGE_PATH")


# Run inference with custom noise
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
#export CUDA_VISIBLE_DEVICES=0

NUM_PROCESSES=$(rp exec 'len(x.split(","))' ---x $CUDA_VISIBLE_DEVICES)
if (( NUM_PROCESSES == 1 )); then
  export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=${MASTER_PORT:-29500}
fi

ic NUM_FRAMES HEIGHT WIDTH CFG_SCALE NUM_INFERENCE_STEPS SEED LORA_DIT LORA_DIT2 HUG_DIR PROMPT OUTPUT INPUT_IMAGE_PATH WARPED_NOISE DEGRADATION_ALPHA CUDA_VISIBLE_DEVICES NUM_PROCESSES

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes $NUM_PROCESSES $([ "$NUM_PROCESSES" -gt 1 ] && echo --multi_gpu) ryan_wan_multigpu_infer_test.py \
    --root "$HUG_DIR/Wan2.2-I2V-A14B" \
    --lora_dit "$LORA_DIT" \
    --lora_dit2 "$LORA_DIT2" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --input_image_path "$INPUT_IMAGE_PATH" \
    --seed "$SEED" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --cfg_scale "$CFG_SCALE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --warped_noise "$WARPED_NOISE" \
    --degradation_alpha "$DEGRADATION_ALPHA"

rp call fansi_print --- "OUTPUT = $OUTPUT" "green green bold italic on dark dark blue"
rp call web_copy_path --- "$OUTPUT"
rp call ntfy_send --- "WAN Inference Done! See $OUTPUT"
