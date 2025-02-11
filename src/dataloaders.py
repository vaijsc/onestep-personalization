from pathlib import Path

from torch.utils.data import Dataset

import numpy as np
import torch
import pdb
import os
import os.path as osp
import json
import random
import glob

def process_res(json_samples, with_out_img=False, type="huy"):
    res = {
        "ref_image": [],
        "text": []
    }

    if with_out_img:
        res["out_image"] = []
    
    for sample in json_samples:
            
        if type == "huy":
            path = "./data/ffhq512_cap/imgs"
            ref_img_path = osp.join(path, sample["img_path"].split("/")[-1])
            
            if with_out_img:
                out_img_path = osp.join(path, sample["img_path"].split("/")[-1])
                res["out_image"].append(out_img_path)
            
            res["ref_image"].append(ref_img_path)
        elif type =="tung":
            res["ref_image"].append(sample["img_path"])
            if with_out_img:
                res["out_image"].append(sample["img_path"])
        res["text"].append(sample["prompt_test"][0])
        
    return res

def get_dict_data(task):
    if task == "coca5k":
        path_json_index = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/bench_data/COCA_dataset/filter_rgb_info_img_cap.json"
        return get_coca_train_task(path_json_index)
    elif task == "coca100k":
        path_json_index = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/bench_data/COCA_dataset_large/unique_info.json"
        return get_coca_train_task(path_json_index)
    elif task == "train_coca100k_val_dreambench":
        return get_data_personalize_task()
    elif task == "train_overfit":
        return get_data_overfit_test()
    elif task == "train_ffhq_cap" or task == "train_ffhq_nested_attn":
        if task == "train_ffhq_cap":
            return get_ffhq_cap()
        elif task == "train_ffhq_nested_attn":
            return get_ffhq_cap(with_s_idx=True)
    elif task == "train_subject200k_nested_attn":
        return get_subject200k()
    
def get_ffhq_cap(ratio_train=0.95, with_s_idx=False):
    def process(json_samples, with_s_idx=False):
        res = {
            "ref_image": [],
            "out_image": [],
            "text": []
        }

        if with_s_idx:
            res["idx_to_alter"] = []
        
        for sample in json_samples:
            res["ref_image"].append(sample["img_path"])
            res["out_image"].append(sample["img_path"])
            res["text"].append(sample["prompt"])
            if with_s_idx:
                res["idx_to_alter"].append(sample["s_index"])
            
        return res
    
    def reformat_json_val(json_val_samples):
        new_json = []
        for sample in json_val_samples:
            for prompt in sample["prompt_test"]:
                new_json.append({
                    "img_path": sample["img_path"],
                    "prompt": prompt,
                    "s_index": sample["s_index"],
                })
 
        
        return new_json

    # path_json_data = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/data/ffhq512_cap/info.json"
    if with_s_idx:
        path_json_data_train = "./data/ffhq512_cap/filtered_info.json"
    else:
        path_json_data_train = "./data/ffhq512_cap/info.json"

    path_json_data_val = "/lustre/scratch/client/movian/research/users/huydnq/research/Data/face_test/info.json"

    with open(path_json_data_train, "r") as fp:
        json_train_samples = json.load(fp)

    with open(path_json_data_val, "r") as fp:
        json_val_samples = json.load(fp)

    json_val_samples = reformat_json_val(json_val_samples)
        
    res_train = process(json_train_samples, with_s_idx=with_s_idx)
    res_val = process(json_val_samples, with_s_idx=with_s_idx)
    
    return res_train, res_val

def get_subject200k(ratio_train=0.95):
    def sub_process(json_in_samples, type="tung"):
        res_samples = {
            "ref_image": [],
            "out_image": [],
            "text": [],
            "idx_to_alter": []
        }
        for sample in json_in_samples:
            if type == "huy":
                path = "../../research/Data/Subjects200K/sep_data/"
                ref_img_path = osp.join(path, sample["reference"].split("sep_data/")[-1])
                out_img_path = osp.join(path, sample["image"].split("sep_data/")[-1])
                
                res_samples["ref_image"].append(ref_img_path)
                res_samples["out_image"].append(out_img_path)
            elif type =="tung":
                res_samples["ref_image"].append(sample["reference"])
                res_samples["out_image"].append(sample["image"])
                
            res_samples["text"].append(sample["prompt"])
            res_samples["idx_to_alter"].append(sample["s_index"])
                
        return res_samples
    
    # path_dreambench_per = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/all_backup_SP/data_test/personalize/simple_test.json"
    path_subject200k_json = "/lustre/scratch/client/movian/research/users/huydnq/research/Data/Subjects200K/dataset.json"
    # path_subject200k_json = "../../research/Data/Subjects200K/dataset.json"

    with open(path_subject200k_json, "r") as fp:
        json_samples = json.load(fp)

    split_train = int(ratio_train*len(json_samples))
    json_samples_train = json_samples[:split_train]
    json_samples_val = json_samples[split_train:]
    
    res_train = sub_process(json_samples_train, type="tung")
    res_val = sub_process(json_samples_val, type="tung")
           
    return res_train, res_val # test overfitting
    
def get_coca_train_task(path_json_index, ratio_train=0.9, with_latent=False):
    with open(path_json_index, "r") as fp:
        json_data = json.load(fp)
        
    # full set
    split_train = int(ratio_train*len(json_data))
    json_train_samples = json_data[:split_train]
    
    json_val_samples = json_data[split_train:]
    
    res_train = process_res(json_train_samples, with_out_img=True)
    res_val = process_res(json_val_samples, with_out_img=True)
    
    return res_train, res_val

def get_data_overfit_test(max_num_prompt_val=8):
    path_dreambench_per = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/all_backup_SP/data_test/personalize/simple_test.json"
    # path_dreambench_per = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/all_backup_SP/data_test/personalize/one_sample.json"

    with open(path_dreambench_per, "r") as fp:
        json_val_samples = json.load(fp)

    res_val = {
        "ref_image": [],
        "out_image": [],
        "text": [],
        "idx_to_alter": []
    }
    
    for sample in json_val_samples:
        max_num_prompt_val = max(max_num_prompt_val, len(sample["prompt_test"]))
        for prompt in sample["prompt_test"][:max_num_prompt_val]:
            res_val["ref_image"].append(sample["img_path"])
            res_val["out_image"].append(sample["img_path"])
            res_val["text"].append(prompt)
            res_val["idx_to_alter"].append(sample["s_index"])
        
    return res_val, res_val # test overfitting

def get_data_personalize_task(max_num_prompt_val=5):
    # train with coca and test with dreambench
    path_coca100k = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/bench_data/COCA_dataset_large/unique_info.json"
    path_dreambench_per = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/data_test/personalize/simple_test.json"

    with open(path_coca100k, "r") as fp:
        json_train_samples = json.load(fp)

    with open(path_dreambench_per, "r") as fp:
        json_val_samples = json.load(fp)

    res_train = process_res(json_train_samples)

    '''
        Process dreambench json format which has following format
        [
            {
                "img_path": "",
                "prompt_test": 
                [
                    prompt1,
                    prompt2,
                    prompt3
                ]
            },
        ]
    '''
    # proccess dreambench json format (check )

    res_val = {
        "ref_image": [],
        "text": []
    }
    
    for sample in json_val_samples:
        for prompt in sample["prompt_test"][:max_num_prompt_val]:
            res_val["ref_image"].append(sample["img_path"])
            res_val["text"].append(prompt)
        
    return res_train, res_val