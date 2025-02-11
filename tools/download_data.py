import os
import wget
import os.path as osp

# URL of the file to download
url = "https://example.com/path/to/your/file"
url = f"https://huggingface.co/datasets/Yuanshi/Subjects200K/resolve/main/data/train-00007-of-00032.parquet?download=true"

output_directory = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/data_personalize/Subject200k/all_parquet_files"


for i in range(8, 32):
    file_id = f"{i:05d}"
    url_donwload = f"https://huggingface.co/datasets/Yuanshi/Subjects200K/resolve/main/data/train-{file_id}-of-00032.parquet?download=true"
    
    print("==========Download file:", f"train-{file_id}-of-00032.parquet")
    output_file = osp.join(output_directory, f"train-{file_id}-of-00032.parquet")
    wget.download(url, out=output_file)
