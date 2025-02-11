import torch
import os.path as osp
import torch.nn.functional as F


from PIL import Image
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image
import sys
sys.path.append("../src")
from utils_src import *
from tqdm import tqdm


if __name__ == "__main__":
    ### for comparing one-step
    all_vis_results = [
        "../vis_results/res_onestep_original",
        "../vis_results/res_onestep_adapter_only",
        "../vis_results/res_onestep_lora+adapter",
        # "../vis_results/res_multistep_original",
        # "../vis_results/res_multistep_nested_attn"
    ]
   
    test = load_dreambench_data()
    # test = load_subject_300k()
    # test = load_ffhq_cap()
    # test = load_
    path_debug_vis = "../debug_vis"
    
    # Define spacing parameters
    whitespace = 10  # Width of whitespace between images
    padding = 10  # Height of padding between rows
    num_channels = 3  # Assuming RGB images
    images_per_row = 10  # Set number of images per row

    for sample in tqdm(test):
        path_ref_img = sample["ref_path"]
        prompt = sample["prompt"]
        
        # for dreambench
        subject_name = path_ref_img.split("/")[-2]
        
        # for ffhq-cap
        # subject_name = path_ref_img.split("/")[-1].split(".")[0]
        
        name_save = f"{subject_name}_{prompt}.png"
        
        # Load the source image
        vis_ref_img = torch.from_numpy(load_512(path_ref_img))\
                        .float().permute(2, 0, 1).unsqueeze(0) / 255
                
        # Initialize the list with the source image
        concat_img = [vis_ref_img]

        for path_edit in all_vis_results:
            res_img_path = osp.join(path_edit, f"{subject_name}_{prompt}.png")
            vis_res_img = torch.from_numpy(load_512(res_img_path))\
                            .float().permute(2, 0, 1).unsqueeze(0) / 255
            concat_img.append(vis_res_img)

        # Create white tensors for whitespace and padding
        height, width = vis_ref_img.shape[2], vis_ref_img.shape[3]
        white_space = torch.ones((1, num_channels, height, whitespace))
        row_padding = torch.ones((1, num_channels, padding, width * images_per_row + whitespace * (images_per_row - 1)))

        # Build rows of 10 images
        rows = []
        for i in range(0, len(concat_img), images_per_row):
            row = concat_img[i]
            for img in concat_img[i + 1:i + images_per_row]:
                row = torch.cat((row, white_space, img), dim=3)  # Concatenate along width
            rows.append(row)

        # Concatenate rows with padding in between
        final_img = rows[0]
        for row in rows[1:]:
            final_img = torch.cat((final_img, row_padding, row), dim=2)  # Concatenate along height

        # Save the final image
        path_save = osp.join(path_debug_vis, name_save)
        save_image(final_img, path_save)