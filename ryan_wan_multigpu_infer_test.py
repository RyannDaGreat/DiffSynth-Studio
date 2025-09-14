#TO RUN: ! CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 ryan_wan_multigpu_infer_test.py

import torch
import argparse
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import torch.distributed as dist
import glob, os

ROOT = "/Wan2.2-I2V-A14B"

# Optional CLI to load LoRA(s)
parser = argparse.ArgumentParser()
parser.add_argument("--lora_dit", type=str, default=None, help="Path(s) to LoRA for dit (comma-separated).")
parser.add_argument("--lora_dit_alpha", type=float, default=1.0, help="Alpha for dit LoRA.")
parser.add_argument("--lora_dit2", type=str, default=None, help="Path(s) to LoRA for dit2 (comma-separated).")
parser.add_argument("--lora_dit2_alpha", type=float, default=1.0, help="Alpha for dit2 LoRA.")
args, _ = parser.parse_known_args()

high_noise_files = sorted(glob.glob(f"{ROOT}/high_noise_model/diffusion_pytorch_model-*.safetensors"))
low_noise_files  = sorted(glob.glob(f"{ROOT}/low_noise_model/diffusion_pytorch_model-*.safetensors"))

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    use_usp=True,   # enable multi-GPU sequence parallel
    model_configs=[
        ModelConfig(path=high_noise_files, offload_device="cpu", skip_download=True),
        ModelConfig(path=low_noise_files,  offload_device="cpu", skip_download=True),
        ModelConfig(path=f"{ROOT}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
        ModelConfig(path=f"{ROOT}/Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
    ],
)

# Load optional LoRAs with existence checks (before VRAM mgmt so adapters get wrapped)
def _load_optional_loras(module, comma_paths, alpha):
    if comma_paths is None:
        return
    for p in [i.strip() for i in comma_paths.split(",") if i.strip()]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"LoRA file not found: {p}")
        pipe.load_lora(module, p, alpha=alpha)

_load_optional_loras(pipe.dit, args.lora_dit, args.lora_dit_alpha)
if getattr(pipe, "dit2", None) is not None:
    _load_optional_loras(pipe.dit2, args.lora_dit2, args.lora_dit2_alpha)

pipe.enable_vram_management()

input_image = Image.open(f"{ROOT}/examples/i2v_input.JPG").resize((832, 480))

video = pipe(
    prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    negative_prompt=("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"

"JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                     "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
    seed=0,
    tiled=True,
    input_image=input_image,
    switch_DiT_boundary=0.9,
)

if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
    save_video(video, "video1.mp4", fps=15, quality=5)
