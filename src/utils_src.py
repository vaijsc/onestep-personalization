import json
import numpy as np
import os
import os.path as osp

from PIL import Image

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def load_personalize_data():
    path_json = "../all_backup_SP/data_test/personalize/simple_test.json"
    with open(path_json, "r") as fp:
        json_data = json.load(fp)
    
    return json_data

def load_quick_test():
    path_json = "../all_backup_SP/data_test/personalize/one_sample.json"

    with open(path_json, "r") as fp:
        json_data = json.load(fp)
        
    new_json = []
    for sample in json_data:
        for prompt in sample["prompt_test"]:
            new_json.append({
                "ref_path": sample["img_path"],
                "prompt": prompt,
                "s_index": sample["s_index"],
            })
        
    return new_json

def load_face_test():
    path_json = "/lustre/scratch/client/movian/research/users/huydnq/research/Data/face_test/info.json"

    with open(path_json, "r") as fp:
        json_data = json.load(fp)
        
    new_json = []
    for sample in json_data:
        for prompt in sample["prompt_test"]:
            new_json.append({
                "ref_path": sample["img_path"],
                "prompt": prompt,
                "s_index": sample["s_index"],
            })
        
    return new_json

def load_subject_300k(type="val"):
    path_json = "/lustre/scratch/client/movian/research/users/huydnq/research/Data/Subjects200K/dataset.json"
    
    with open(path_json, "r") as fp:
        json_data = json.load(fp)
        
    new_json = []
    for sample in json_data:
        new_json.append({
            "ref_path": sample["reference"],
            "prompt": sample["prompt"],
            "s_index": sample["s_index"],
        })
        
    if type == "val":
        return new_json[-100:] # last 100 samples
        
    return new_json

def load_dreambench_data(prompt_idx_chose=0, max_samples=None):
    path_json = "/lustre/scratch/client/movian/research/users/huydnq/research/Data/dreambench/test_dataset.json"
    with open(path_json, "r") as fp:
        json_data = json.load(fp)
    
    new_json_data = []
    for idx, sample in enumerate(json_data):
        prompt_list = sample["prompts"]
        
        for prompt in prompt_list:
            new_json_data.append({
                "ref_path": sample["ref_path"],
                "prompt": prompt
            })
    
    if max_samples is not None:
        num_samples = min(max_samples, len(new_json_data))
    else:
        num_samples = len(new_json_data)
    return new_json_data[:num_samples]

def load_ffhq_cap():
    path_json = "../data/ffhq512_cap/info.json"
    with open(path_json, "r") as fp:
        json_data = json.load(fp)
        
    new_json = []
    for sample in json_data:
        new_json.append({
            "ref_path": sample["img_path"],
            "prompt": sample["prompt_test"][0]
        })
    
    return new_json[-100:] # last 100 samples
        
def load_sample_reconstruct_coca(size="5k", option="train"):
    if size == "5k":
        path_json = "/lustre/scratch/client/scratch/research/group/khoigroup/quangnh24/SBE/bench_data/COCA_dataset/filter_rgb_info_img_cap.json"
    
    ratio_train = 0.9
    with open(path_json, "r") as fp:
        json_data = json.load(fp)
        
    split_train = int(ratio_train*len(json_data))
    if option == "train":
        return json_data[:split_train]
    elif option == "test":
        return json_data[split_train:]
    
def load_512(image_path: str, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3]    
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image