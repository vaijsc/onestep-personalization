import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
)
from huggingface_hub import hf_hub_download
from torchvision.utils import save_image
import os
from PIL import Image
from utils_src import *
from tqdm import tqdm

base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "tianweiy/DMD2"
ckpt_name = "model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000/pytorch_model.bin"
ip_adapter_model = "h94/IP-Adapter"
# ip_adapter_model = "/lustre/scratch/client/vinai/users/huydnq/research/1-step-personalize/Experiments/pretrained"
path_save_debug = "../debug_vis"

@torch.no_grad()
def main():
    # Load model.
    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(
        "cuda", torch.float16
    )
    unet.load_state_dict(
        torch.load(hf_hub_download(repo_name, ckpt_name), map_location="cuda")
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16",
        safety_checker = None,
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        ip_adapter_model,
        subfolder="models",
        weight_name="ip-adapter-plus_sd15.bin",
    )

    generator = torch.Generator(device="cuda")

    test = load_sample_reconstruct_coca(option="train")[:20]
    scale_vary = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for sample in tqdm(test):
        src_p = sample["src_p"]
        img_path = sample["img_path"]

        input_img = load_512(img_path)

        pil_input = Image.open(img_path)

        vis_src_img = torch.from_numpy(input_img).float().permute(2, 0, 1).unsqueeze(0) / 255
        vis_src_img = vis_src_img.to("cuda")
        
        vis_img = [vis_src_img]
        for scale in scale_vary:
            pipe.set_ip_adapter_scale(scale)
            out_img = pipe(
                src_p,
                ip_adapter_image=pil_input,
                guidance_scale=0, 
                generator=generator,
                num_inference_steps=1,
            ).images[0]

            out_img = torch.from_numpy(np.array(out_img)).permute(2, 0, 1) / 255
            out_img = out_img.unsqueeze(0).to("cuda")
            vis_img.append(out_img)
        
        vis_img = torch.cat(vis_img)
        path_save = osp.join(path_save_debug, f"{src_p}.png")
        save_image(vis_img, path_save)   

if __name__ == "__main__":
    main()