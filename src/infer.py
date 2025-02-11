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
from ip_adapter import IPAdapter
from tqdm import tqdm

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
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

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class MyIPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, name):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.name = name

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_adapter(self, ckpt_adapter):
        # new stands for trained model while original stands for original ip-adapter weight
        if "train" in ckpt_adapter or "debug" in ckpt_adapter:
            type = "new"
        else:
            type = "original"
        
        sd = torch.load(ckpt_adapter, map_location="cpu")
        image_proj_sd = {}
        ip_sd = {}
        
        if type == "new":
            lora_sd = {}
        
        for k in sd:
            if k.startswith("unet"):
                if "lora" in ckpt_adapter:
                    prefix = "unet."
                    new_key = k[len(prefix):]
                    lora_sd[new_key] = sd[k]
                else:
                    pass
            elif k.startswith("image_proj"):
                if type == "new":
                    subkey1 = k.split(".")[1] 
                    subkey2 = k.split(".")[2] 
                    
                    image_proj_sd[f"{subkey1}.{subkey2}"] = sd[k]
                else:
                    image_proj_sd = sd[k]
            elif k.startswith("adapter_modules") or k.startswith("ip_adapter"):
                if type == "new":
                    subkey1 = k.split(".")[1] 
                    subkey2 = k.split(".")[2] 
                    subkey3 = k.split(".")[3]
                    
                    ip_sd[f"{subkey1}.{subkey2}.{subkey3}"] = sd[k]
                else:
                    ip_sd = sd[k]

        self.image_proj_model.load_state_dict(image_proj_sd)
        self.adapter_modules.load_state_dict(ip_sd)
        
        if "lora" in ckpt_adapter:
            self.unet.load_lora_adapter(
                lora_sd
            )

    def set_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

class OSPersonalize():
    def __init__(self, path_pretrained_sb_unet,
                 path_pretrained_adapter,
                 image_encoder_path="h94/IP-Adapter",
                 pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", 
                 num_ip_tokens=4, 
                 weight_dtype=torch.float32, device="cuda"):
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
            path_pretrained_sb_unet, 
            subfolder="unet_ema"
        ).to(self.device)
        
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
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
                                                cross_attention_dim=cross_attention_dim,
                                                num_tokens=self.num_ip_tokens)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(self.unet.attn_processors.values()).to(self.device)
        
        return MyIPAdapter(self.unet, image_proj_model, adapter_modules, name="model")
        
    def get_x0_from_noise(self, sample, model_output):

        timestep = torch.ones((1,), dtype=torch.int64, device=self.device)
        timestep = timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        alpha_t = (alphas_cumprod[timestep] ** 0.5).view(-1, 1, 1, 1)
        sigma_t = ((1 - alphas_cumprod[timestep]) ** 0.5).view(-1, 1, 1, 1)

        pred_original_sample = (sample - sigma_t * model_output) / alpha_t
        return pred_original_sample
    
    @torch.no_grad()
    def gen_img(self,
                ref_img_path, prompt,
                bs=1):
        
        prompt_embeds = encode_prompt([prompt], self.text_encoder, self.tokenizer)["prompt_embeds"].to(device=self.device)
        input_shape = (1, 4, 64, 64)
        # noise = torch.randn(input_shape, dtype=self.weight_dtype, device=self.device)
        noise = gen_random_tensor_fix_seed(input_shape, 1, self.weight_dtype, self.device)

        pil_ref_image = [Image_PIL.open(ref_img_path)]
        clip_images = [self.clip_image_processor(images=sample, return_tensors="pt").pixel_values for sample in pil_ref_image]
        clip_images = torch.cat(clip_images, dim=0)
        image_embeds = self.image_encoder(clip_images.to(self.device, dtype=self.weight_dtype)).image_embeds
        
        # Get denoised image with one step image generator (sbv2 in this case) with ip-condition (scale=1)
        noise_pred = self.generator_with_adapter(noise, self.last_timestep, prompt_embeds, image_embeds)
        pred_latents = self.get_x0_from_noise(
            noise, noise_pred
        )
        
        out_image = pred_latents / self.vae.config.scaling_factor
        out_image = (
            self.vae.decode(out_image.to(dtype=self.weight_dtype)).sample.float() + 1
        ) / 2
        
        return out_image

def infer_exp1():
    # path_save_infer = "../debug_vis"
    # path_save_infer = "../vis_results/res_onestep_original"
    # path_save_infer = "../vis_results/res_onestep_adapter_only"
    # path_save_infer = "../vis_results/res_onestep_lora+adapter"
    path_save_infer = "../vis_results/res_onestep_lora+adapter_80000"
    
    makedirs(path_save_infer)
    
    path_sb_unet = "../all_backup_SP/ckpt/sbv2_sd1.5/0.7"
    
    # path_sb_adapter = "../all_backup_SP/ckpt/ip-adapter_sd15.bin"
    # path_sb_adapter = "../tmp_results/train_ddips_both_sb_ffhq_cap/checkpoint-20000/ip_adapter.bin"
    # path_sb_adapter = "../tmp_results/train_ddips_both_sb_with_lora_ffhq_cap/checkpoint-20000/ip_adapter.bin"
    path_sb_adapter = "../tmp_results/train_ddips_both_sb_with_lora_ffhq_cap/checkpoint-80000/ip_adapter.bin"

    # load samples
    # samples = [
    #     {
    #         "ref_img": "../all_backup_SP/data_test/personalize/reference_image/me.jpg",
    #         "text": "A man with a dog"
    #     }
    # ]
    
    # samples = load_ffhq_cap()
    # samples = load_dreambench_data()
    samples = load_subject_300k()
    
    # load pretrained one-step model
    os_personalize_model = OSPersonalize(path_sb_unet, path_sb_adapter)
    
    for sample in tqdm(samples):
        prompt = sample["prompt"]
        ref_img_path = sample["ref_path"]
        
        # for dreambench
        subject_name = ref_img_path.split("/")[-2]
        
        # for ffhq-cap
        # subject_name = ref_img_path.split("/")[-1].split(".")[0]
            
        vis_ref_img = torch.from_numpy(
            load_512(ref_img_path)
        ).float().permute(2, 0, 1).unsqueeze(0) / 255
        
        vis_ref_img = vis_ref_img.to("cuda")
        
        res_img = os_personalize_model.gen_img(ref_img_path, prompt)

        # vis_img = torch.cat([vis_ref_img, res_img_new, res_img_original])
        # vis_img = torch.cat([vis_ref_img, res_img_original])
        # visualize and debug
        path_save = osp.join(path_save_infer, f"{subject_name}_{prompt}.png")
        save_image(res_img, path_save)

@torch.no_grad()
def infer_multistep_resampling_ddip(
    type_sampling="ddips",
    NUM_SAMPLES = 1,
    NUM_INFERENCE_STEPS = 50,
    BS = 1,
    SEED = 20,
    CFG_IMG = 7.5,
    CFG_PROMPT = 7.5
):
    def prepare_embeds(pipe, ip_model, image, prompt):
        # prepare image-prompt embeds
        image_prompt_embeds, uncond_image_prompt_embeds = ip_model.get_image_embeds(
            pil_image=image, clip_image_embeds=None
        )

        # uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds).to(device)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, NUM_SAMPLES, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * NUM_SAMPLES, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, NUM_SAMPLES, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * NUM_SAMPLES, seq_len, -1)

        ## prepare prompt embeds
        input_id = tokenize_captions(pipe.tokenizer, [prompt] * NUM_SAMPLES).to("cuda")
        prompt_embeds = pipe.text_encoder(input_id)[0]

        null_prompt_ids = tokenize_captions(pipe.tokenizer, [""] * NUM_SAMPLES).to("cuda")
        null_prompt_embeds = pipe.text_encoder(null_prompt_ids)[0]

        uncond_img_uncond_prompt_embeds = torch.cat([null_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        cond_img_uncond_prompt_embeds = torch.cat([null_prompt_embeds, image_prompt_embeds], dim=1)
        uncond_img_cond_prompt_embeds = torch.cat([prompt_embeds, uncond_image_prompt_embeds], dim=1)
        cond_img_cond_prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)

        return uncond_img_uncond_prompt_embeds,\
            uncond_img_cond_prompt_embeds, \
            cond_img_uncond_prompt_embeds, \
            cond_img_cond_prompt_embeds
    
    path_save_vis = osp.join("../vis_results", f"res_multistep_{type_sampling}")
    # path_save_vis = "../debug_vis"
    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "h94/IP-Adapter"
    ip_ckpt = "../all_backup_SP/ckpt/ip-adapter_sd15.bin"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    ip_model.pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = ip_model.pipe.scheduler.timesteps
    generator = get_generator(SEED, ip_model.pipe.device)
    num_channels_latents = ip_model.pipe.unet.config.in_channels

    # load samples here
    samples = load_dreambench_data()
    # samples = load_subject_300k()
    
    
    # samples = [{
    #     "ref_path": "/lustre/scratch/client/movian/research/users/huydnq/research/Data/dreambench/dog2/02.jpg",
    #     "prompt": "a dog wearing pink glasses"
    # }]
    
    for sample in tqdm(samples):
        ref_img_path = sample["ref_path"]
        prompt = sample["prompt"]
        subject_name = ref_img_path.split("/")[-2]
        
        # read image prompt
        image = Image.open(ref_img_path)
        image.resize((512, 512))
        
        # prepare image embeds
        embeds1, embeds2, embeds3, embeds4 = prepare_embeds(pipe, ip_model, image, prompt)
        
        ## prepare prompt embeds
        input_id = tokenize_captions(pipe.tokenizer, [prompt] * NUM_SAMPLES).to("cuda")
        prompt_embeds = pipe.text_encoder(input_id)[0]
        
        # sampling
        latents = ip_model.pipe.prepare_latents(
            BS * NUM_SAMPLES,
            num_channels_latents,
            512,
            512,
            prompt_embeds.dtype,
            ip_model.pipe.device,
            generator,
            None,
        )
        
        for i, t in enumerate(tqdm(timesteps)):

            latent_model_input = latents
            latent_model_input = ip_model.pipe.scheduler.scale_model_input(latent_model_input, t)

            noise_pred_prompt = ip_model.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds2,
                return_dict=False,
            )[0]

            noise_pred_img = ip_model.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds3,
                return_dict=False,
            )[0]

            noise_pred_uncond = ip_model.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds1,
                return_dict=False,
            )[0]

            noise_pred_full = ip_model.pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds4,
                return_dict=False,
            )[0]
            
            if type_sampling == "ddips":
                # perform guidance with ddips score
                noise_pred = noise_pred_uncond + CFG_IMG * (noise_pred_img - noise_pred_uncond) \
                                                + CFG_PROMPT * (noise_pred_prompt - noise_pred_uncond)
                                                    
            else:
                # perform guidance with traditional score
                noise_pred = noise_pred_uncond + CFG_IMG * (noise_pred_full - noise_pred_uncond)
            
            latents = ip_model.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        images = ip_model.pipe.vae.decode(latents / ip_model.pipe.vae.config.scaling_factor, return_dict=False)[0]
        images = (images + 1) / 2
        
        name_save = f"{subject_name}_{prompt}.png"
        path_save = osp.join(path_save_vis, name_save)
        
        save_image(images, path_save)

    
if __name__ == "__main__":
    infer_exp1()
    # infer_multistep_resampling_ddip(type_sampling="ddips")
    # infer_multistep_resampling_ddip(type_sampling="original")
