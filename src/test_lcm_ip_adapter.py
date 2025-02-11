from diffusers import StableDiffusionPipeline, LCMScheduler
import torch
from diffusers.utils import load_image

model_id =  "sd-dreambooth-library/herge-style"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

list_scale = [0, 0.2, 0.4, 0.6, 0.8, 1]
num_steps = [1, 2, 4, 8, 16]

for step in num_steps:
    for scale in list_scale:
        pipe.set_ip_adapter_scale(scale)

        prompt = "best quality, high quality"
        image = load_image("https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png")
        images = pipe(
            prompt=prompt,
            ip_adapter_image=image,
            num_inference_steps=step,
            guidance_scale=1,
        ).images[0]

        images.save(f"../debug_vis/{step}_{scale}.png")