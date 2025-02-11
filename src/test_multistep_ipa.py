import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image

from ip_adapter import IPAdapter, IPAdapterPlus

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
ip_ckpt = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15.bin"
# ip_ckpt = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/ckpt/ip-adapter_sd15_light.bin"
device = "cuda"
num_samples_gen = 5

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

# generate image variations
images = ip_model.generate(pil_image=image, 
                           prompt=["a dog wearing pink sunglasses"],
                           num_samples=num_samples_gen, num_inference_steps=50,
                           scale=0.75)
grid = image_grid(images, 1, num_samples_gen)
grid.save("../test_multistep_ipa.png")  