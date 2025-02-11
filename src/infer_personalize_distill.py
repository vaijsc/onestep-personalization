import torch
import pdb
import numpy as np
import json
import os.path as osp
import sys
import torch.nn.functional as F
import numpy as np
import random


from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, UNet2DModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image


from utils_src import *
# from utils_operator import *
# from utils_sam import *
from tqdm import tqdm

from ip_adapter.ip_adapter import ImageProjModel
from transformers import CLIPImageProcessor
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
    from ip_adapter.attention_processor import IPAttnProcessor2_0WithIPMaskController, AttnProcessor2_0WithMaskController
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

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

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

def create_generator(checkpoint_path, base_model=None):
    if base_model is None:
        generator = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet"
        ).float()
        generator.requires_grad_(False)
    else:
        generator = base_model

    # sometime the state_dict is not fully saved yet 
    counter = 0
    while True:
        try:
            state_dict = torch.load(checkpoint_path)
            break 
        except:
            print(f"fail to load checkpoint {checkpoint_path}")

            counter += 1 

            if counter > 100:
                return None

    # # unwrap the generator 
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     if k.startswith("feedforward_model."):
    #         new_state_dict[k[len("feedforward_model."):]] = v

    # print(generator.load_state_dict(new_state_dict, strict=True))
    print(generator.load_state_dict(state_dict, strict=True))
    return generator

def get_x0_from_noise(sample, model_output, timestep):
    # alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    # 0.0047 corresponds to the alphas_cumprod of the last timestep (999)
    alpha_prod_t = (torch.ones_like(timestep).float() * 0.0047).reshape(-1, 1, 1, 1) 
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample

class AuxiliaryModel():
    def __init__(self, 
                 model_name="runwayml/stable-diffusion-v1-5",
                 image_encoder_path="h94/IP-Adapter", device="cuda"):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        ).to(device, dtype=torch.float32)
        self.image_encoder.requires_grad_(False)
        
        self.clip_image_processor = CLIPImageProcessor()

class IPDMD2Model(torch.nn.Module):
    def __init__(self, path_dmd2, path_ckpt_ip,
                 aux_model, device="cuda",
                 num_ip_token=4,
                 is_direct_plugged=False):
        super().__init__()
        self.unet = create_generator(path_dmd2)
        self.unet.to(device).eval()
        self.device = device
        self.aux_model = aux_model

        self.timestep = torch.ones((1,), dtype=torch.int64, device=device)
        self.timestep = self.timestep * (self.aux_model.noise_scheduler.config.num_train_timesteps - 1)

        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.aux_model.image_encoder.config.projection_dim,
            clip_extra_context_tokens=num_ip_token,
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
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_ip_token).to(device)
                attn_procs[name].load_state_dict(weights)
                
        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values())
    
        # prepare stuff
        alphas_cumprod = self.aux_model.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[self.timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
        
        if path_ckpt_ip is not None:
            if is_direct_plugged:
                self.load_from_original_checkpoint(path_ckpt_ip)
            else:
                self.load_state_dict(torch.load(path_ckpt_ip))

    def load_from_original_checkpoint(self, path_ckpt_ip: str):
        sd = torch.load(path_ckpt_ip, map_location="cpu")
        image_proj_sd = {}
        ip_sd = {}
        for k in sd:
            if k.startswith("unet"):
                pass
            elif k.startswith("image_proj"):
                image_proj_sd = sd[k]
            elif k.startswith("ip_adapter"):
                ip_sd = sd[k]

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
            bs = len(prompts)
            noise = torch.cat([noise] * bs, dim=0)
        
        one_timestep = torch.ones(1, device=self.device, dtype=torch.long)
        final_timestep = torch.ones((1,), dtype=torch.int64, device="cuda") * 999

        model_pred = self.unet(noise, final_timestep, prompt_embeds).sample
        pred_original_sample = get_x0_from_noise(
            noise, model_pred, one_timestep
        )

        pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
        image = (
            self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
        ) / 2
        
        noise_image = noise / self.aux_model.vae.config.scaling_factor
        noise_image = (
            self.aux_model.vae.decode(noise_image.to(dtype=self.aux_model.vae.dtype)).sample.float() + 1
        ) / 2

        return image, noise_image

@torch.no_grad()
def test_dmd2_distill_with_pretrained():
    aux_model = AuxiliaryModel()
    path_save_edit_img = "../debug_vis"
    test = load_personalize_data()
    
    path_ckpt_dmdv2 = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/checkpoints/dmdv2/sd15/pytorch_model.bin"
    
    path_ckpt_ip = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_distill_ip_dmd2_coca100k/checkpoint-5000"
    ip_model_path = osp.join(path_ckpt_ip, "ip_adapter.bin")
    ip_dmd_model = IPDMD2Model(path_ckpt_dmdv2, ip_model_path, 
                              aux_model)

    # ip_model_path = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15.bin"
    # ip_dmd_model = IPDMD2Model(path_ckpt_dmdv2, ip_model_path, 
    #                           aux_model, is_direct_plugged=True)

    for sample in tqdm(test):
        prompt_list = sample["prompt_test"]
        ref_img_path = sample["img_path"]
        ref_pil_input = Image.open(ref_img_path)

        vis_ref_img = torch.from_numpy(load_512(ref_img_path)).float().permute(2, 0, 1).unsqueeze(0) / 255
        vis_ref_img = vis_ref_img.to("cuda")

        for personalized_prompt in prompt_list:

            scale_vary = [0, 0.4, 0.8, 1.2, 1.6, 2.0]

            vis_img = [vis_ref_img]
            for pick_s in scale_vary:
                output_img, _ = ip_dmd_model.gen_img(pil_image=ref_pil_input, 
                                                    prompts=[personalized_prompt],
                                                    scale=pick_s)
                
                vis_img.append(output_img)
            
            vis_img = torch.cat(vis_img)
            path_save = osp.join(path_save_edit_img, f"{personalized_prompt}.png")
            save_image(vis_img, path_save)

@torch.no_grad()
def test_dmd2_distill_from_scratch():
    aux_model = AuxiliaryModel()
    path_save_edit_img = "../debug_vis"
    test = load_personalize_data()
    
    path_ckpt_dmdv2 = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/checkpoints/dmdv2/sd15/pytorch_model.bin"
    # path_ckpt_ip = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/all_ckpt/train_distill_ip_dmd2_from_scratch_coca100k/checkpoint-5000"
    # ip_model_path = osp.join(path_ckpt_ip, "ip_adapter.bin")
    
    ip_model_path = None
    ip_dmd_model = IPDMD2Model(path_ckpt_dmdv2, ip_model_path, 
                              aux_model, num_ip_token=32)

    for sample in tqdm(test):
        prompt_list = sample["prompt_test"]
        ref_img_path = sample["img_path"]
        ref_pil_input = Image.open(ref_img_path)

        vis_ref_img = torch.from_numpy(load_512(ref_img_path)).float().permute(2, 0, 1).unsqueeze(0) / 255
        vis_ref_img = vis_ref_img.to("cuda")

        for personalized_prompt in prompt_list:

            scale_vary = [0, 0.4, 0.8, 1.2, 1.6, 2.0]

            vis_img = [vis_ref_img]
            for pick_s in scale_vary:
                output_img, _ = ip_dmd_model.gen_img(pil_image=ref_pil_input, 
                                                    prompts=[personalized_prompt],
                                                    scale=pick_s)
                
                vis_img.append(output_img)
            
            vis_img = torch.cat(vis_img)
            path_save = osp.join(path_save_edit_img, f"{personalized_prompt}.png")
            save_image(vis_img, path_save)
        
if __name__ == "__main__":
    test_dmd2_distill_with_pretrained()
    # test_dmd2_distill_from_scratch()