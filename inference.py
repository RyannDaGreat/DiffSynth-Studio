import rp
from rp.git.CommonSource.noise_warp import mix_new_noise
import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from transformers import CLIPVisionModel
from einops import rearrange

device=rp.select_torch_device(prefer_used=True,reserve=True)
dtype=torch.bfloat16

if not 'pipe' in vars():
    # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=dtype)

    lora_path = '/efs/users/jordanlin/public/ryan/CleanCode/Github/finetrainers/untracked/outputs/wani2v_GWTF_rank1024/lora_weights/002929/pytorch_lora_weights.safetensors'
    pipe.load_lora_weights(lora_path)

    pipe.to(device)
    
T,H,W=49,480,832
LT = (T-1) // 4 + 1

noises_path=f'/root/CleanCode/Sandbox/wan_gwtf_test/cut_and_drag_cat_{H}x{W}/noises.npy'
video_path = '/root/CleanCode/Sandbox/wan_gwtf_test/cat-climbing-in-the-tree-compressed.jpg_copy1.mp4'
degradation = 1

noises = rp.omni_load(noises_path)
noises = rp.resize_list(noises, LT)
noises = mix_new_noise(noises,alpha=degradation)
noises = torch.tensor(noises,device=device,dtype=dtype)
noises = rearrange(noises, "T H W C -> C T H W")  # Weird wan thing

image = rp.load_image(video_path, use_cache=True)
image = rp.cv_resize_image(image, (H, W))
image = rp.as_pil_image(image)


#Kept from original demo, can probably be simplified
max_area = H * W
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    #"An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    #"the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    #"An elephant falls on an astronaut"
    "A cat climbs down a tree, ultrarealistic detail and breath-taking movie-like camera shot."
    #"An astronaut does a backflip High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


####################

num_inference_steps=20
num_frames=49
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    num_inference_steps=num_inference_steps,
    width=width,
    num_frames=num_frames,
    guidance_scale=5.0,
    latents=noises[None],    
).frames[0]
output_path = f"wan_gwtf_test__deg={degradation}__steps={num_inference_steps}__T={num_frames}"
print(rp.save_video_mp4(output,output_path,framerate=20))
