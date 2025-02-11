import torch
import numpy as np
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image


from ip_adapter import IPAdapter, IPAdapterPlus
from ip_adapter.utils import is_torch2_available, get_generator

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "h94/IP-Adapter"
ip_ckpt = "../all_backup_SP/ckpt/ip-adapter_sd15.bin"
# ip_ckpt = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15_light.bin"
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

# read image prompt
image = Image.open("/lustre/scratch/client/vinai/users/huydnq/research/1-step-personalize/Data/dreambench/dog2/02.jpg")
image.resize((256, 256))

# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
# ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device)


'''
    Ancestral sampling with refined score
'''
NUM_PROMPTS = 1
NUM_SAMPLES = 5
NUM_INFERENCE_STEPS = 30
BS = 1
PROMPT = "a dog wearing pink sunglasses"
SEED = 20
CFG_IMG = 7.5
CFG_PROMPT = 7.5


## prepare image-prompt embeds
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
input_id = tokenize_captions(pipe.tokenizer, [PROMPT] * NUM_SAMPLES).to("cuda")
prompt_embeds = pipe.text_encoder(input_id)[0]

null_prompt_ids = tokenize_captions(pipe.tokenizer, [""] * NUM_SAMPLES).to("cuda")
null_prompt_embeds = pipe.text_encoder(null_prompt_ids)[0]

uncond_img_uncond_prompt_embeds = torch.cat([null_prompt_embeds, uncond_image_prompt_embeds], dim=1)
cond_img_uncond_prompt_embeds = torch.cat([null_prompt_embeds, image_prompt_embeds], dim=1)
uncond_img_cond_prompt_embeds = torch.cat([prompt_embeds, uncond_image_prompt_embeds], dim=1)
cond_img_cond_prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)


# combined_embeds = torch.cat([uncond_img_uncond_prompt_embeds,
#                              cond_img_uncond_prompt_embeds,
#                              uncond_img_cond_prompt_embeds], dim=0)

## sampling
ip_model.pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
timesteps = ip_model.pipe.scheduler.timesteps
generator = get_generator(SEED, ip_model.pipe.device)

num_channels_latents = ip_model.pipe.unet.config.in_channels
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

with torch.no_grad():
    for i, t in enumerate(tqdm(timesteps)):

        latent_model_input = latents
        latent_model_input = ip_model.pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred_prompt = ip_model.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=uncond_img_cond_prompt_embeds,
            return_dict=False,
        )[0]

        noise_pred_img = ip_model.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond_img_uncond_prompt_embeds,
            return_dict=False,
        )[0]

        noise_pred_uncond = ip_model.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=uncond_img_uncond_prompt_embeds,
            return_dict=False,
        )[0]

        noise_pred_full = ip_model.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=cond_img_cond_prompt_embeds,
            return_dict=False,
        )[0]


        # # perform guidance
        # noise_pred = noise_pred_uncond + CFG_IMG * (noise_pred_img - noise_pred_uncond) \
        #                                 + CFG_PROMPT * (noise_pred_prompt - noise_pred_uncond) 

        noise_pred = noise_pred_uncond + CFG_IMG * (noise_pred_img - noise_pred_uncond) \
                                        + CFG_PROMPT * (noise_pred_prompt - noise_pred_uncond) \

        # noise_pred = noise_pred_uncond + 7.5 * (noise_pred_prompt - noise_pred_uncond) 
        # noise_pred = noise_pred_uncond + 7.5 * (noise_pred_img - noise_pred_uncond) 

        # noise_pred = noise_pred_uncond + CFG_IMG * (noise_pred_full - noise_pred_prompt) \
        #                                 + CFG_PROMPT * (noise_pred_prompt - noise_pred_img) 

        # noise_pred_full = ip_model.pipe.unet(
        #     latent_model_input,
        #     t,
        #     encoder_hidden_states=cond_img_cond_prompt_embeds,
        #     return_dict=False,
        # )[0]

        # perform guidance
        # noise_pred = noise_pred_uncond + CFG_IMG * (noise_pred_full - noise_pred_uncond)
        # noise_pred = rescale_noise_cfg(noise_pred, noise_pred_full, guidance_rescale=CFG_IMG)
        
        latents = ip_model.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

images = ip_model.pipe.vae.decode(latents / ip_model.pipe.vae.config.scaling_factor, return_dict=False)[0]
images = (images + 1) / 2
save_image(images, "../test_multistep_ipa_resampling.png", nrow=NUM_SAMPLES)
