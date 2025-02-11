import torch
import pdb
import numpy as np
import json
import os.path as osp
import sys
import torch.nn.functional as F
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import random

from tqdm import tqdm
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, UNet2DModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image
from diffusers.utils import load_image, make_image_grid
from torchvision.utils import make_grid, draw_bounding_boxes
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from sklearn.manifold import TSNE
from scipy.ndimage import binary_dilation

from ip_adapter.ip_adapter import ImageProjModel
from transformers import CLIPImageProcessor
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from utils_src import *

from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
    from ip_adapter.attention_processor import IPAttnProcessor2_0WithIPMaskController, AttnProcessor2_0WithMaskController
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class SBV2Model():
    def __init__(self, path_ckpt_sbv2, model_name="stabilityai/stable-diffusion-2-1-base"):
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to("cuda")
        self.unet.eval()

        self.timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.timestep = self.timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        
        
        # prepare stuff
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
    
    @torch.no_grad()
    def gen_img(self, prompts, noise=None, 
                return_input_noise=False, 
                return_decoded_latent=False,
                return_model_pred=False):
        if noise is None:
            bs = len(prompts)
            noise = torch.randn(bs, 4, 64, 64, device="cuda")
        
        input_id = tokenize_captions(self.tokenizer, prompts).to("cuda")
        encoder_hidden_state = self.text_encoder(input_id)[0]
        
        model_pred = self.unet(noise, self.timestep, encoder_hidden_state).sample

        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
        if self.noise_scheduler.config.thresholding:
            pred_original_sample = self.noise_scheduler._threshold_sample(
                pred_original_sample
            )
        elif self.noise_scheduler.config.clip_sample:
            clip_sample_range = self.noise_scheduler.config.clip_sample_range
            pred_original_sample = pred_original_sample.clamp(
                -clip_sample_range, clip_sample_range
            )

        pred_original_sample = pred_original_sample / self.vae.config.scaling_factor
        image = (
            self.vae.decode(pred_original_sample).sample + 1
        ) / 2
        
        noise_image = noise / self.vae.config.scaling_factor
        noise_image = (
            self.vae.decode(noise_image.to(dtype=self.vae.dtype)).sample.float() + 1
        ) / 2

        if return_model_pred:
            return image, model_pred
        if return_decoded_latent and return_input_noise:
            return image, noise_image, noise, pred_original_sample * self.vae.config.scaling_factor
        if return_input_noise:
            return image, noise_image, noise
        if return_decoded_latent:
            return image, pred_original_sample * self.vae.config.scaling_factor
        return image, noise_image

class AuxiliaryModel():
    def __init__(self, 
                 model_name="stabilityai/stable-diffusion-2-1-base",
                 image_encoder_path="h94/IP-Adapter", device="cuda"):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        self.clip_image_processor = CLIPImageProcessor()
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        ).to(device, dtype=torch.float32)
        self.image_encoder.requires_grad_(False)

class IPSBV2Model(torch.nn.Module):
    def __init__(self, path_ckpt_sbv2, path_ckpt_ip,
                 aux_model, device="cuda"):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to(device)
        self.unet.eval()
        self.device = device
        self.aux_model = aux_model

        self.timestep = torch.ones((1,), dtype=torch.int64, device=device)
        self.timestep = self.timestep * (self.aux_model.noise_scheduler.config.num_train_timesteps - 1)

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.aux_model.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        ).to(device)        
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
                attn_procs[name] = AttnProcessor().to(device)
            else:
                # this is for cross-attention
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(device)
                attn_procs[name].load_state_dict(weights)
                
        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
    
        # prepare stuff
        alphas_cumprod = self.aux_model.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
        
        # self.load_state_dict(torch.load(path_ckpt_ip))
    
    def load_ip_adapter(self, path_ckpt_ip):
        
        sd = torch.load(path_ckpt_ip, map_location="cpu")
        image_proj_sd = {}
        ip_sd = {}
        for k in sd:
            if k.startswith("unet"):
                pass
            elif k.startswith("image_proj_model"):
                image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            elif k.startswith("adapter_modules"):
                ip_sd[k.replace("adapter_modules.", "")] = sd[k]

        self.image_proj_model.load_state_dict(image_proj_sd)
        self.adapter_modules.load_state_dict(ip_sd)
        
    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.aux_model.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.aux_model.image_encoder(clip_image.to(self.device, dtype=torch.float32)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float32)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        return image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor) or isinstance(attn_processor, IPAttnProcessor2_0WithIPMaskController):
                attn_processor.scale = scale
                
    @torch.no_grad()
    def gen_img(self, pil_image=None,
                prompts=None, noise=None,
                scale=1.,):
        
        self.set_scale(scale)
            
        num_samples = len(prompts)
        
        if prompts is None:
            prompts = ["best quality, high quality"]
        
        if isinstance(pil_image, list):
            # mixing pil image condition
            image_prompt_embeds_1 = self.get_image_embeds(
                pil_image=pil_image[0]
            )
            image_prompt_embeds_2 = self.get_image_embeds(
                pil_image=pil_image[1]
            )
            
            image_prompt_embeds = (image_prompt_embeds_1 + image_prompt_embeds_2) / 2 
        else:
            image_prompt_embeds = self.get_image_embeds(
                pil_image=pil_image
            )
            
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        
        with torch.inference_mode():
            input_id = tokenize_captions(self.aux_model.tokenizer, prompts).to("cuda")
            prompt_embeds_ = self.aux_model.text_encoder(input_id)[0]
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        
        if noise is None:
            bs = len(prompts)
            noise = torch.randn(bs, 4, 64, 64, device="cuda")
        else:
            # bs = len(prompts)
            # noise = torch.cat([noise] * bs, dim=0)
            
            # for testing idea mix random noise only
            bs = len(prompts)
            noise = torch.cat([noise] * bs, dim=0)
        
        model_pred = self.unet(noise, self.timestep, prompt_embeds).sample

        if model_pred.shape[1] == noise.shape[1] * 2:
            model_pred, _ = torch.split(model_pred, noise.shape[1], dim=1)

        pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
        
        if self.aux_model.noise_scheduler.config.thresholding:
            pred_original_sample = self.aux_model.noise_scheduler._threshold_sample(
                pred_original_sample
            )
        elif self.aux_model.noise_scheduler.config.clip_sample:
            clip_sample_range = self.aux_model.noise_scheduler.config.clip_sample_range
            pred_original_sample = pred_original_sample.clamp(
                -clip_sample_range, clip_sample_range
            )

        pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
        image = (
            self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
        ) / 2

        return image    

def gen_random_tensor_fix_seed(input_shape, seed, weight_dtype, device):
    # Set the random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Ensure that all operations are deterministic on the current device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generate the random tensor
    return torch.randn(*input_shape, dtype=weight_dtype, device=device)

@torch.no_grad()
def test_infer_os_personalize():
    test = load_sample_reconstruct_coca(option="train")[:20]

    path_ckpt_sbv2 = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/sb_v2_ckpt/0.5"
    # path_ckpt_ip_model = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_ip_sb2_coca100k/checkpoint-70000"
    path_ckpt_ip_model = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_ip_elpips_sb2_with_vsd_coca100k/bk_ckpt_10000"
    # path_ckpt_ip_model = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_ip_elpips_sb2_coca100k/checkpoint-90000"


    path_save_debug = "../debug_vis"

    ip_model_path = osp.join(path_ckpt_ip_model, "ip_adapter.bin")
    aux_model = AuxiliaryModel()
    ip_sb_model = IPSBV2Model(path_ckpt_sbv2, ip_model_path, 
                              aux_model)
    
    scale_vary = [0, 0.2, 0.4, 0.6, 0.8, 1]
    
    for sample in tqdm(test):
        src_p = sample["src_p"]
        
        img_path = sample["img_path"]
        input_img = load_512(img_path)

        processed_image = torch.from_numpy(input_img).float().permute(2, 0, 1) / 255
        processed_image = processed_image.unsqueeze(0).to("cuda", dtype=torch.float32)
        pil_input = Image.open(img_path)
        
        vis_src_img = torch.from_numpy(input_img).float().permute(2, 0, 1).unsqueeze(0)/255
        vis_src_img = vis_src_img.to("cuda")

        # noise = torch.randn(1, 4, 64, 64, device="cuda")
        noise = gen_random_tensor_fix_seed((1, 4, 64, 64), 0, torch.float32, "cuda")

        vis_img = [vis_src_img]
        for scale in scale_vary:
            out_img = ip_sb_model.gen_img(pil_input, prompts=[src_p], noise=noise, scale=scale)

            vis_img.append(out_img)

        vis_img = torch.cat(vis_img)
        path_save = osp.join(path_save_debug, f"{src_p}.png")
        save_image(vis_img, path_save)    

if __name__ == "__main__":
    test_infer_os_personalize()


