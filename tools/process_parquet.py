import pyarrow.parquet as pq
import io
import os.path as osp
import json

from PIL import Image
from tqdm import tqdm

output_img_path = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/data/ffhq512_cap/imgs"
output_json_file = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/data/ffhq512_cap/info.json"

json_list = []
for i in tqdm(range(0, 54)):

    # Path to the Parquet file
    input_file_path = f"/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/data/ffhq512_cap/{i}.parquet"
    print(f"Processing {input_file_path}")

    # Read the Parquet file
    table = pq.read_table(input_file_path)
    df = table.to_pandas()
    len_rows = len(df['image'])

    for idx in tqdm(range(len_rows)):
        img_name = df['image'][idx]['path']
        txt = df['text'][idx]
        byte_img_data = df['image'][idx]['bytes']
        
        image = Image.open(io.BytesIO(byte_img_data))

        # Save the image to a file
        img_path = osp.join(output_img_path, img_name)
        image.save(img_path)

        json_list.append(
            {
                "img_path": img_path,
                "prompt_test": [txt]
            }
        )
    
with open(output_json_file, "w") as fp:
    json.dump(json_list, fp, indent=4)
