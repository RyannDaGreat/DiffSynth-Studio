# TO RUN (example):
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 ryan_wan_multigpu_infer_test.py

import torch
import fire
from typing import Optional
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import torch.distributed as dist
import glob, os

import rp

def main(
    root: str = "/Wan2.2-I2V-A14B",
    lora_dit: Optional[str] = None,
    lora_dit2: Optional[str] = None,
    input_image_path: Optional[str] = None,
    output: str = "video1.mp4",
    fps: int = 15,
    quality: int = 5,
    prompt: str = (
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    ),
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
        "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    seed: int = 0,
    tiled: bool = True,
    switch_DiT_boundary: float = 0.9,
    noise_file: Optional[str] = None,
    degradation: float = 0.0,
):
    """Run multi-GPU inference with optional LoRAs for DiT and DiT2.

    Args:
        root: Base directory of WAN model assets.
        lora_dit: LoRA file path for DiT. Alpha is fixed to 1.0.
        lora_dit2: LoRA file path for DiT2. Alpha is fixed to 1.0.
        input_image_path: Path to the input image. Defaults to {root}/examples/i2v_input.JPG
        output: Output video filename.
        fps: Frames per second for output video.
        quality: Video quality (lower is higher quality for some codecs).
        prompt: Positive prompt text.
        negative_prompt: Negative prompt text.
        seed: Random seed.
        tiled: Whether to enable tiling.
        switch_DiT_boundary: Switch boundary between DiT stages.
        noise_file: Path to custom noise .npy file. Leave None for random noise.
        degradation: Degradation level (0.0=pure custom noise, 1.0=pure random).
    """

    high_noise_files = sorted(glob.glob(f"{root}/high_noise_model/diffusion_pytorch_model-*.safetensors"))
    low_noise_files = sorted(glob.glob(f"{root}/low_noise_model/diffusion_pytorch_model-*.safetensors"))

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=True,  # enable multi-GPU sequence parallel
        model_configs=[
            ModelConfig(path=high_noise_files, offload_device="cpu", skip_download=True),
            ModelConfig(path=low_noise_files, offload_device="cpu", skip_download=True),
            ModelConfig(path=f"{root}/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
            ModelConfig(path=f"{root}/Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
        ],
    )

    # Load optional LoRA(s) before VRAM management so adapters get wrapped
    if lora_dit:
        lora_path = lora_dit.strip()
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
        rp.fansi_print(f'RANK {dist.get_rank()}: Loaded pipe.dit LoRA {lora_path}','green')
    if lora_dit2 and getattr(pipe, "dit2", None) is not None:
        lora2_path = lora_dit2.strip()
        if not os.path.isfile(lora2_path):
            raise FileNotFoundError(f"LoRA file not found: {lora2_path}")
        pipe.load_lora(pipe.dit2, lora2_path, alpha=1.0)
        rp.fansi_print(f'RANK {dist.get_rank()}: Loaded pipe.dit2 LoRA {lora_path}','green')

    pipe.enable_vram_management()

    if input_image_path is None:
        input_image_path = f"{root}/examples/i2v_input.JPG"
    # input_image = Image.open(input_image_path).resize((832, 480))
    input_image = rp.load_image(input_image_path)
    input_image = rp.cv_resize_image(input_image,(480,832))
    input_image = rp.as_byte_image(input_image)
    input_image = rp.as_rgb_image(input_image)
    input_image = rp.as_pil_image(input_image)


    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        tiled=tiled,
        input_image=input_image,
        switch_DiT_boundary=switch_DiT_boundary,
        custom_noise_file=noise_file,
        degradation_level=degradation,
    )

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        save_video(video, output, fps=fps, quality=quality)


if __name__ == "__main__":
    fire.Fire(main)
