import rp


@rp.memoized
def load_model(model_path):
    cached_model_path = rp.get_cache_file_path(model_path)
    rp.copy_file(model_path, cached_model_path, show_progress=True, resume=True)
    return rp.load_safetensors(cached_model_path)


####

model_a_path = "models/train/Wan2.2-I2V-A14B_high_noise_lora_GWTF_Dev_Deepspeed_<T=81>/step-250.safetensors"
model_b_path = "models/train/Wan2.2-I2V-A14B_high_noise_lora_GWTF_Dev_Deepspeed_<T=81>/step-40000.safetensors"
model_a = load_model(model_a_path)
model_b = load_model(model_b_path)
alpha = 0.85
model_z = {
    key: rp.blend(val_a, val_b, alpha)
    for key, (val_a, val_b) in rp.eta(rp.dict_zip(model_a, model_b), "Blending LoRAs")
}

rp.save_safetensors(model_z, f"merge_models_test/alpha={alpha}.safetensors")
