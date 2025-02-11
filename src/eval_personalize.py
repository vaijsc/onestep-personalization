import time
import torch
import clip
import json
import os
import io
import os.path as osp
import argparse
import pandas as pd
import pdb


from tqdm import tqdm
from PIL import Image
from scipy import spatial
from torchvision.transforms import transforms
from utils_src import *


def encode(image, model, transform, metric=None, DEVICE="cuda"):
    image_input = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if metric == 'clip_i':
            image_features = model.encode_image(image_input).detach().cpu().float()
        elif metric == 'dino':
            image_features = model(image_input).detach().cpu().float()
    return image_features

def cal_personalize_score(ref_img_path, res_img_path,
                          model_dino, transform_dino,
                          model_clip, transform_clip,
                          prompt, DEVICE="cuda"):

    # cal clip-i score
    generated_features = encode(Image.open(res_img_path).convert('RGB'), 
                                model_clip, transform_clip, metric='clip_i')
                             
    gt_features = encode(Image.open(ref_img_path).convert('RGB'), 
                         model_clip, transform_clip, metric='clip_i')

    clip_i_score = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                 gt_features.view(gt_features.shape[1]))

    if clip_i_score > 1 or clip_i_score < -1:
        raise ValueError(" strange similarity value")

    # cal dino score 
    generated_features = encode(Image.open(res_img_path).convert('RGB'), 
                                model_dino, transform_dino, metric='dino')
    gt_features = encode(Image.open(ref_img_path).convert('RGB'), 
                         model_dino, transform_dino, metric='dino')
    dino_score = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                 gt_features.view(gt_features.shape[1]))

    if dino_score > 1 or dino_score < -1:
            raise ValueError(" strange similarity value")
        
    # cal clip-text score
    
    generated_features = encode(Image.open(res_img_path).convert('RGB'), 
                                model_clip, transform_clip, metric='clip_i')
    
    text_features = clip.tokenize(prompt).to(DEVICE)
    with torch.no_grad():
        text_features = model_clip.encode_text(text_features).detach().cpu().float()
    
    clip_t_score = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                    text_features.view(text_features.shape[1]))
    
    return clip_i_score, clip_t_score, dino_score


if __name__ == "__main__":
    # define inputs
    path_res_personalize = ""
    all_dict_path = {
        # "multistep_ddips_sampling": "../vis_results/res_multistep_ddips",
        # "multistep_IP": "../vis_results/res_multistep_original",
        # "res_nested_attention_70k": "../vis_results/res_multistep_nested_attn",
        # "res_nested_attention_120k": "../vis_results/res_multistep_nested_attn_120k",
        # "res_nested_attention_190k": "../vis_results/res_multistep_nested_attn_190k",
        # "1step_original": "../vis_results/res_onestep_original",
        # "1step_adapter_only": "../vis_results/res_onestep_adapter_only",
        # "1step_lora+adapter": "../vis_results/res_onestep_lora+adapter"
        "1step_lora+adapter_80k": "../vis_results/res_onestep_lora+adapter_80000"
        
    }
    # DEVICE = "cuda:0"
    DEVICE = "cuda"
    result_df = pd.DataFrame()
    
    # define models
    model_clip, transform_clip = clip.load("../all_backup_SP/ckpt/clip/ViT-B-32.pt", DEVICE)
    model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model_dino.eval()
    model_dino.to(DEVICE)
    transform_dino = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # load sample test
    # test_samples = load_dreambench_data()
    # test_samples = load_ffhq_cap()
    test_samples = load_subject_300k()
    
    total_clip_i = 0.0000
    total_clip_t = 0.0000
    total_dino = 0.0000
    
    for method_name in all_dict_path:
        path_res_personalize = all_dict_path[method_name]
        for sample in tqdm(test_samples):
            path_ref_img = sample["ref_path"]
            prompt = sample["prompt"]
            
            # for dreambench
            subject_name = path_ref_img.split("/")[-2]
            
            # for ffhq-cap
            # subject_name = path_ref_img.split("/")[-1].split(".")[0]
            
            name_save = f"{subject_name}_{prompt}.png"
            path_res_img = osp.join(path_res_personalize, name_save)
            
            (score_clip_i, score_clip_t, score_dino) = cal_personalize_score(path_ref_img, path_res_img,
                                                                    model_dino, transform_dino,
                                                                    model_clip, transform_clip,
                                                                    prompt)    
            
            total_clip_i += score_clip_i
            total_clip_t += score_clip_t
            total_dino += score_dino    
        
        total_clip_i /= len(test_samples)
        total_clip_t /= len(test_samples)
        total_dino /= len(test_samples)
        
        metrics_of_method = {
            "Method": method_name,
            "CLIP-I": total_clip_i,
            "DINO": total_dino,
            "CLIP-T": total_clip_t,
        }
    
        result_df = pd.concat(
            [result_df, pd.DataFrame([metrics_of_method])], ignore_index=True
        )
    
    result_df.to_csv(f"../debug_vis/scores_personalize.csv", index=False)

        
    