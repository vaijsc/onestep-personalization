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
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        
        self.clip_image_processor = CLIPImageProcessor()

class TeacherIPA(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, name):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.name = name

    def load_from_checkpoint(self, ckpt_path: str):
        
        sd = torch.load(ckpt_path, map_location="cpu")
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

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

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
                scale=1.,
                return_latent=False):
        
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

        if return_latent:
            return pred_original_sample
        else:
            pred_original_sample = pred_original_sample / self.aux_model.vae.config.scaling_factor
            image = (
                self.aux_model.vae.decode(pred_original_sample.to(dtype=torch.float32)).sample.float() + 1
            ) / 2
            
            noise_image = noise / self.aux_model.vae.config.scaling_factor
            noise_image = (
                self.aux_model.vae.decode(noise_image.to(dtype=self.aux_model.vae.dtype)).sample.float() + 1
            ) / 2
            return image, noise_image
        
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
        return image, pred_original_sample * self.vae.config.scaling_factor

@torch.no_grad()
def decode_latents(latents: torch.Tensor, vae):
    latents = latents / vae.config.scaling_factor
    image = (
        vae.decode(latents.to(dtype=torch.float32)).sample.float() + 1
    ) / 2

    return image

def get_ipa_generator(unet_gen_ip, 
                      image_encoder,
                      name_model,
                      num_ip_tokens=4, device="cuda"):
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet_gen_ip.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=num_ip_tokens,
    ).to(device)
    
    # init adapter modules
    attn_procs = {}
    unet_sd = unet_gen_ip.state_dict()
    for name in unet_gen_ip.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet_gen_ip.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet_gen_ip.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet_gen_ip.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet_gen_ip.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                               cross_attention_dim=cross_attention_dim,
                                               num_tokens=num_ip_tokens)
            attn_procs[name].load_state_dict(weights)
            attn_procs[name] = attn_procs[name].to(device)
    unet_gen_ip.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet_gen_ip.attn_processors.values())
    
    generator_with_adapter = TeacherIPA(unet_gen_ip, image_proj_model, adapter_modules, name=name_model)

    return generator_with_adapter

def optimize_latent(init_latent, personalized_prompt, 
                    teacher_ipa, aux_model, ref_pil_input,
                    device="cuda",
                    ITER=150,
                    CFG=20,
                    option="dual_guide"): # CFG=20
    
    init_latent = init_latent.detach()
    optimize_latent = init_latent.clone().requires_grad_() 
    optimizer = torch.optim.SGD([optimize_latent], lr=5000) #10000 # 20000
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.9)

    # prepare prompt emebd
    input_id = tokenize_captions(aux_model.tokenizer, [personalized_prompt]).to("cuda")
    prompt_embeds = aux_model.text_encoder(input_id)[0]

    null_prompt_ids = tokenize_captions(aux_model.tokenizer, [""]).to("cuda")
    null_prompt_embeds = aux_model.text_encoder(null_prompt_ids)[0]

    # prepare image embed
    clip_image = aux_model.clip_image_processor(images=ref_pil_input, return_tensors="pt").pixel_values
    clip_image_embeds = aux_model.image_encoder(clip_image.to(device, dtype=torch.float32)).image_embeds
    image_embeds = teacher_ipa.image_proj_model(clip_image_embeds)
    null_image_embeds = torch.zeros_like(image_embeds).to(device)

    bs_embed, seq_len, _ = image_embeds.shape
    image_embeds = image_embeds.repeat(1, 1, 1)
    image_embeds = image_embeds.view(bs_embed, seq_len, -1)

    combined_hidden_states = torch.cat([prompt_embeds, image_embeds], dim=1)
    combined_hidden_states_2 = torch.cat([prompt_embeds, null_image_embeds], dim=1)
    combined_hidden_states_3 = torch.cat([null_prompt_embeds, image_embeds], dim=1)
    combined_hidden_states_4 = torch.cat([null_prompt_embeds, null_image_embeds], dim=1)

    list_optimized_image = [decode_latents(optimize_latent, aux_model.vae)]

    timesteps_range = torch.tensor([0, 1]) * aux_model.noise_scheduler.config.num_train_timesteps

    with tqdm(total=len(range(ITER)), file=sys.stdout) as pbar:
        for epoch in range(ITER):
            optimizer.zero_grad()

            # add noise to latents
            noise = torch.randn_like(optimize_latent)

            timesteps = torch.randint(*timesteps_range.long(), (1,), device=device).long()
            noisy_latents = aux_model.noise_scheduler.add_noise(optimize_latent, noise, timesteps).to(device)

            teacher_pred_cond_img_cond_text = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states).sample
            teacher_pred_uncond_img_cond_text = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states_2).sample
            teacher_pred_cond_img_uncond_text = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states_3).sample
            teacher_pred_uncond_img_uncond_text = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states_4).sample

            # '''
            #     Idea 1
            # '''
            # teacher_pred_with_ip = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states).sample
            # teacher_pred_with_null_ip = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states_4).sample

            # dip_gradient =  teacher_pred_with_ip - teacher_pred_with_null_ip

            # '''
            #     Idea 2
            # '''
            # teacher_pred_with_ip = teacher_ipa.unet(noisy_latents, timesteps, combined_hidden_states).sample
            # dip_gradient =  teacher_pred_with_ip - noise

            # exp 1
            # score = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_uncond_text)
            # exp 2
            # score = teacher_pred_cond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_cond_img_uncond_text)
            # exp 3
            # score = teacher_pred_uncond_img_cond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_cond_text)
            # exp 4
            # score = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_uncond_text - teacher_pred_uncond_img_uncond_text)
            # # exp 5
            # score = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_uncond_img_cond_text - teacher_pred_uncond_img_uncond_text)
            # exp 6
            # score = teacher_pred_cond_img_cond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_uncond_text)
            # exp 7

            score4 = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_uncond_text - teacher_pred_uncond_img_uncond_text)
            score5 = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_uncond_img_cond_text - teacher_pred_uncond_img_uncond_text)

            if option == "dual_guide":
                '''
                    Idea 1: Naive combination
                '''
                # score = score4 + score5 - noise
                # score = score4 + score5
                # dip_gradient =  score - noise

                '''
                    Idea 1.*: anh khoi thread
                '''
                # score = score4 + score5 - noise
                # score = score4 + score5
                # dip_gradient =  score - noise
                score = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_uncond_text)
                dip_gradient = score - teacher_pred_uncond_img_cond_text

                '''
                    Idea 1.1: similar to naive combination but follow compositional visual gen paper (see form 11)
                '''
                # score_img = CFG * (teacher_pred_cond_img_uncond_text - teacher_pred_uncond_img_uncond_text)
                # score_text = CFG * (teacher_pred_uncond_img_cond_text - teacher_pred_uncond_img_uncond_text) # this does not work
                # score = teacher_pred_uncond_img_uncond_text + score_img + score_text
                # dip_gradient = score - noise

                '''
                    Idea 1.2: naive combine all four score
                '''
                # score1 = teacher_pred_cond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_cond_img_uncond_text)
                # score2 = teacher_pred_uncond_img_cond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_cond_text)
                # # dip_gradient = score1 - noise + score2 - noise + score4 - noise + score5 - noise
                # # dip_gradient = score1 - noise + score2 - noise
                # dip_gradient = score4 - noise + score5 - noise

                '''
                    Idea 2
                '''
                # score = teacher_pred_uncond_img_uncond_text + CFG * (
                #     teacher_pred_cond_img_uncond_text + teacher_pred_uncond_img_cond_text - 2 * teacher_pred_uncond_img_uncond_text
                # )
                # dip_gradient =  score - noise

                '''
                    Idea 3: Combine
                '''
                # score_img = score4
                # score_text = score5
                # score_joint = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_uncond_text)

                # dip_gradient = score_img + score_text - 2 * noise

                '''
                    Idea 4: Similar to ip2p, check collaborative score distillation
                '''
                # score_img = CFG * (teacher_pred_cond_img_uncond_text - teacher_pred_uncond_img_uncond_text)
                # # score_text = CFG * (teacher_pred_cond_img_cond_text - teacher_pred_cond_img_uncond_text) # this does not work
                # score_text = CFG * (teacher_pred_uncond_img_cond_text - teacher_pred_uncond_img_uncond_text) 

                # ## increase cfg for each score amplify the importance of each modality in the generated output

                # dip_gradient =  teacher_pred_uncond_img_uncond_text + score_img + score_text - noise
                # # dip_gradient = teacher_pred_uncond_img_uncond_text + score_text - noise
                # dip_gradient = teacher_pred_cond_img_uncond_text + score_text - noise

            elif option == "img_guide":
                score = score4
                dip_gradient =  score - noise
            elif option == "img_guide_edit":
                dip_gradient =  teacher_pred_cond_img_uncond_text - teacher_pred_uncond_img_uncond_text
            elif option == "text_guide":
                score = score5
                dip_gradient = score - noise
            elif option == "text_guide_edit":
                # score6 = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_uncond_text)
                # score7 = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_uncond_text - teacher_pred_uncond_img_uncond_text)

                '''
                    Maybe this works: run over several case to double check
                '''
                score6 = teacher_pred_uncond_img_cond_text # with score 5, it does not work
                score7 = teacher_pred_uncond_img_uncond_text

                '''
                    Another test: does not work
                '''
                # score6 = score5
                # score7 = noise

                '''
                    Another test: This does not work
                '''
                # score6 = teacher_pred_cond_img_cond_text
                # score7 = teacher_pred_cond_img_uncond_text

                '''
                    DDS: idea
                '''
                # score6 = teacher_pred_cond_img_cond_text
                # score7 = teacher_pred_cond_img_uncond_text

                dip_gradient =  score6 - score7

            elif option == "simple_guide":
                score = teacher_pred_uncond_img_uncond_text + CFG * (teacher_pred_cond_img_cond_text - teacher_pred_uncond_img_uncond_text)
                dip_gradient = score - noise

            # dip_gradient =  score - noise
            dip_gradient = torch.nan_to_num(dip_gradient)

            loss = 0.5 * F.mse_loss(optimize_latent, (optimize_latent - dip_gradient).detach(), reduction="mean")
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()

            optimized_image = decode_latents(optimize_latent, aux_model.vae)
            list_optimized_image.append(optimized_image)

            pbar.set_description('train loss: %0.6f' % (loss))
            pbar.update(1)

    for i in range(len(list_optimized_image)):
        path_progress_vis = osp.join("../debug_vis", f"progress-{i}-{option}.png")

        save_image(list_optimized_image[i], path_progress_vis)
    
    return optimize_latent

def test_optimize_dip(device="cuda", seed=2):
    path_teacher_unet = "runwayml/stable-diffusion-v1-5"
    path_original_ipa = "../all_backup_SP/ckpt/ip-adapter_sd15.bin"

    aux_model = AuxiliaryModel()

    # prepare teacher IPA
    teacher_unet = UNet2DConditionModel.from_pretrained(
        path_teacher_unet, subfolder="unet"
    ).to(device)
    teacher_unet.eval()
    teacher_ipa = get_ipa_generator(teacher_unet, 
                                    aux_model.image_encoder, 
                                    "teacher")
    
    teacher_ipa.load_from_checkpoint(path_original_ipa)

    # prepare one-step t2i generator
    # path_ckpt_dmdv2 = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/checkpoints/dmdv2/sd15/pytorch_model.bin"
    # ip_dmd2_model = IPDMD2Model(path_ckpt_dmdv2, None, 
    #                           aux_model)

    # sbv2 gen (backbone sd2.1)
    # path_ckpt_sbv2 = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/sb_v2_ckpt/0.5"
    # sbv2_model = SBV2Model(path_ckpt_sbv2)

    # sbv2 gen (backbone sd1.5)
    path_ckpt_sbv2 = "../all_backup_SP/ckpt/sbv2_sd1.5/0.7"
    sbv2_model = SBV2Model(path_ckpt_sbv2, model_name="runwayml/stable-diffusion-v1-5")
    
    path_save_edit_img = "../debug_vis"
    test = [load_personalize_data()[0]]

    fix_noise = gen_random_tensor_fix_seed((1, 4, 64, 64), seed,
                                           torch.float32, "cuda")

    prompt_gen = "A photo of a white dog"

    # for sample in tqdm(test):
    for sample in test:
        prompt_list = sample["prompt_test"]
        ref_img_path = sample["img_path"]
        ref_pil_input = Image.open(ref_img_path)

        vis_ref_img = torch.from_numpy(load_512(ref_img_path)).float().permute(2, 0, 1).unsqueeze(0) / 255
        vis_ref_img = vis_ref_img.to("cuda")

        vis_img = [vis_ref_img]
        for personalized_prompt in prompt_list:
            # first get gen image by one-step model with prompt
            # init_latent = ip_dmd2_model.gen_img(pil_image=ref_pil_input, 
            #                                     prompts=["A photo of a white dog"],
            #                                     scale=0, noise=fix_noise,
            #                                     return_latent=True) # no image condition, normal gen

            personalized_prompt = "A dog wearing wizard hat"

            _, init_latent = sbv2_model.gen_img([prompt_gen], fix_noise)

            '''
                Single optimization
            '''
            # do optimization with latents initialized from output_img
            # optimized_latent = optimize_latent(init_latent, personalized_prompt, 
            #                                    teacher_ipa, aux_model, ref_pil_input,
            #                                    CFG=20, option="dual_guide")

            # random_latent = gen_random_tensor_fix_seed((1, 4, 64, 64), 0,
            #                                torch.float32, "cuda")

            # # # init latent is random image
            # optimized_latent = optimize_latent(random_latent, personalized_prompt, 
            #                                    teacher_ipa, aux_model, ref_pil_input)

            # '''
            #     Two-stage optimization: optimize image first, then text edit
            # '''
            # optimized_latent1 = optimize_latent(init_latent, personalized_prompt, 
            #                                    teacher_ipa, aux_model, ref_pil_input, option="img_guide")


            # start with reference image
            img_path = sample["img_path"]
        
            input_img = load_512(sample["img_path"])
            # processed_image = torch.from_numpy(input_img).float().permute(2, 0, 1) / 127.5 - 1
            processed_image = torch.from_numpy(input_img).float().permute(2, 0, 1) / 255
            processed_image = processed_image.unsqueeze(0).to("cuda", dtype=torch.float32) * 2 - 1    

            optimized_latent1 = sbv2_model.vae.encode(processed_image).latent_dist.sample()
            optimized_latent1 = optimized_latent1 * sbv2_model.vae.config.scaling_factor                               
            
            optimized_latent2 = optimize_latent(optimized_latent1, personalized_prompt, 
                                               teacher_ipa, aux_model, ref_pil_input, option="text_guide_edit")

            '''
                Two-stage optimization: optimize text first, then image edit
            '''
            # optimized_latent1 = optimize_latent(init_latent, personalized_prompt, 
            #                                    teacher_ipa, aux_model, ref_pil_input, option="text_guide")
            # optimized_latent2 = optimize_latent(optimized_latent1, personalized_prompt, 
            #                                    teacher_ipa, aux_model, ref_pil_input, option="img_guide_edit")
            
            breakpoint()


if __name__ == "__main__":
    test_optimize_dip(seed=2) # seed = 2 