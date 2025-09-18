# Copy Model To This Machine (Super Fast the Second Time)
HUG_DIR=/huggingface_models
mkdir -p $HUG_DIR
rclone copy --progress --transfers 128 /root/CleanCode/Github/DiffSynth-Studio/huggingface_models $HUG_DIR

#Icecream equivalent for bash
ic() { for v in "$@"; do echo "[ic] $v=${!v}"; done; }

# Define LoRA checkpoints from rp call download_to_cache
LORA_DIT=$( rp call download_to_cache --- "models/train/Wan2.2-I2V-A14B_high_noise_lora/step-8700.safetensors" --show_progress True)
LORA_DIT2=$(rp call download_to_cache --- "models/train/Wan2.2-I2V-A14B_low_noise_lora/step-4000.safetensors"  --show_progress True)

# Choose the content
PROMPT="An ultra-detailed 360 equirectangular panorama inside an aquarium with many colorful fish swimming, crystal-clear water, reflections and caustics, immersive viewpoint, cinematic lighting"
OUTPUT="fish_aquarium_360.mp4"
INPUT_IMAGE_PATH="aquarium_360_video.mp4"

# Choose the content
PROMPT="a car driving torwards the camera on a desert as the camera zooms at high speed with the car driving through the desert"
OUTPUT="car_360_video.mp4"
INPUT_IMAGE_PATH='https://static.vecteezy.com/system/resources/thumbnails/049/023/305/small_2x/hdri-360-panorama-near-white-car-on-roadside-of-gravel-dusty-road-in-field-with-blue-sky-and-awesome-clouds-in-equirectangular-full-seamless-spherical-projection-for-vr-ar-content-photo.jpg'

# Choose the content
PROMPT="the camera drone flies through the city torwards the buildings quickly and swirls around a driving black car thats zooming down the street. the camera spins around 360"
OUTPUT="city_car_360_video.mp4"
INPUT_IMAGE_PATH='Hitrovskaya_square_PANO_20150910_183647.jpg'

OUTPUT=$(rp call get_unique_copy_path --- "$OUTPUT")
INPUT_IMAGE_PATH=$(rp call download_to_cache --- "$INPUT_IMAGE_PATH")

ic LORA_DIT LORA_DIT2 HUG_DIR PROMPT OUTPUT INPUT_IMAGE_PATH

# Run inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --multi_gpu ryan_wan_multigpu_infer_test.py \
    --root "$HUG_DIR/Wan2.2-I2V-A14B" \
    --lora_dit "$LORA_DIT" \
    --lora_dit2 "$LORA_DIT2" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --input_image_path "$INPUT_IMAGE_PATH"