import torch
import random
import numpy as np
import os.path as osp

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from transformers import AutoTokenizer, PretrainedConfig
from PIL import Image as Image_PIL
from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from ip_adapter.ip_adapter import ImageProjModel
from torchvision.utils import save_image
from ip_adapter.utils import is_torch2_available
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from utils_src import *
from ip_adapter.utils import get_generator
from tqdm import tqdm

from ip_adapter.nested_attention_processor import NestedAttnProcessor2_0 as NestedAttnProcessor, AttnProcessor2_0 as AttnProcessor
from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor

def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_adapter(self, ckpt_path: str):
        
        sd = torch.load(ckpt_path, map_location="cpu")
        image_proj_sd = {}
        ip_sd = {}
        for k in sd:
            if k.startswith("unet"):
                pass
            elif k.startswith("image_proj"):
                subkey1 = k.split(".")[1] 
                subkey2 = k.split(".")[2] 
                
                image_proj_sd[f"{subkey1}.{subkey2}"] = sd[k]
            elif k.startswith("adapter_modules") or k.startswith("ip_adapter"):
                subkey1 = k.split(".")[1] 
                subkey2 = k.split(".")[2] 
                subkey3 = k.split(".")[3]
                
                ip_sd[f"{subkey1}.{subkey2}.{subkey3}"] = sd[k]

        self.image_proj_model.load_state_dict(image_proj_sd)
        self.adapter_modules.load_state_dict(ip_sd)

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, NestedAttnProcessor):
                attn_processor.scale = scale

    def register_indices_to_alter(self, indices_to_alter):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, NestedAttnProcessor):
                attn_processor.indices_to_alter = indices_to_alter


class NestedIPMultistep:
    def __init__(self,
                 path_pretrained_adapter,
                 image_encoder_path="h94/IP-Adapter",
                 pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", 
                 num_ip_tokens=4, 
                 weight_dtype=torch.float32, device="cuda"):
        self.weight_dtype = torch.float32
        self.device = device
        
        self.num_ip_tokens = num_ip_tokens
        self.weight_dtype = weight_dtype
        self.device = device
        self.last_timestep = torch.ones((1,), dtype=torch.int64, device=self.device) * 999
    
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
        )

        # Load stuff
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device, dtype=torch.float32)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        ).to(self.device, dtype=torch.float32)

        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, subfolder="models/image_encoder").to(self.device)
        self.image_encoder.requires_grad_(False)
        
        self.clip_image_processor = CLIPImageProcessor()
        
        self.generator_with_adapter = self.plug_adapter()
        self.generator_with_adapter.load_adapter(path_pretrained_adapter)
    
    def plug_adapter(self):
        
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_ip_tokens,
        ).to(self.device)
        
        # init adapter modules
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]

                weights = {
                    "to_k_nest.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_nest.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = NestedAttnProcessor(hidden_size=hidden_size, 
                                                cross_attention_dim=cross_attention_dim,
                                                num_tokens=self.num_ip_tokens)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values()).to(self.device)
        
        generator_with_adapter = IPAdapter(self.unet, image_proj_model, adapter_modules)

        return generator_with_adapter

    def generate(self,
                 prompt, index_to_alter,
                 image_path,
                 NUM_STEPS=50,
                 CFG=7.5):
        
        self.noise_scheduler.set_timesteps(NUM_STEPS, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        prompt_embeds = encode_prompt([prompt], self.text_encoder, self.tokenizer)["prompt_embeds"].to(self.device)
        null_prompt_embeds = encode_prompt([""], self.text_encoder, self.tokenizer)["prompt_embeds"].to(self.device)

        # Sampling noise
        input_shape = (1, 4, 512 // 8, 512 // 8)
        noise = torch.randn(input_shape, dtype=self.weight_dtype, device=self.device)
        indices_to_alter = torch.tensor([index_to_alter]).unsqueeze(1)

        latents = noise

        # Get image embeds for IP-Adapter
        pil_ref_image = [Image_PIL.open(image_path)]
                        
        clip_images = [self.clip_image_processor(images=sample, return_tensors="pt").pixel_values for sample in pil_ref_image]
        clip_images = torch.cat(clip_images, dim=0)
        clip_image_embeds = self.image_encoder(clip_images.to(self.device, dtype=self.weight_dtype)).image_embeds
        null_image_embeds = torch.zeros_like(clip_image_embeds).to(self.device)

        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            self.generator_with_adapter.register_indices_to_alter(indices_to_alter)

            noise_pred_uncond = self.generator_with_adapter(latent_model_input, t, null_prompt_embeds, null_image_embeds)
            noise_pred = self.generator_with_adapter(latent_model_input, t, prompt_embeds, clip_image_embeds)

            noise_pred = noise_pred_uncond + CFG * (noise_pred - noise_pred_uncond) 
            latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        out_image = latents / self.vae.config.scaling_factor
        out_image = (
            self.vae.decode(out_image.to(dtype=self.weight_dtype)).sample.float() + 1
        ) / 2
        
        return out_image

@torch.no_grad()
def infer():
    # load model
    # path_adapter = "../tmp_results/train_multistep_nested_attention_subject200k/checkpoint-70000/ip_adapter.bin"
    # path_adapter = "../tmp_results/train_multistep_nested_attention_subject200k/checkpoint-120000/ip_adapter.bin"
    path_adapter = "../tmp_results/train_multistep_nested_attention_subject200k/checkpoint-190000/ip_adapter.bin"
    
    # path_save = "../debug_vis"
    # path_save = "../vis_results/res_multistep_nested_attn_120k"
    path_save = "../vis_results/res_multistep_nested_attn_190k"
    
    makedirs(path_save)
    
    model = NestedIPMultistep(path_adapter)
    
    # load samples
    samples = load_subject_300k()
    # samples = load_dreambench_data()
    
    for sample in samples:
        prompt = sample["prompt"]
        idx_to_alter = sample["s_index"]
        path_ref_img = sample["ref_path"]
        subject_name = path_ref_img.split("/")[-2]
        
        res_img = model.generate(prompt, idx_to_alter, path_ref_img)
        
        # vis_ref_img = torch.from_numpy(
        #     load_512(path_ref_img)
        # ).float().permute(2, 0, 1).unsqueeze(0) / 255
        
        # vis_ref_img = vis_ref_img.to("cuda")
        # res_img = torch.cat([vis_ref_img, res_img])
        
        path_save_res_img = osp.join(path_save, f"{subject_name}_{prompt}.png")
        save_image(res_img, path_save_res_img)

if __name__ == "__main__":
    infer()