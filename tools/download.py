import os
import requests

# Base URL
base_url = "https://huggingface.co/api/datasets/Ryan-sjtu/ffhq512-caption/parquet/default/train/{index}.parquet"

# Directory to save files
output_dir = "/lustre/scratch/client/movian/research/users/tungnt132/SwiftPersonalize/data/ffhq512_cap"
os.makedirs(output_dir, exist_ok=True)

# Download files
for index in range(0, 1):  # From 1 to 52 inclusive
    file_url = base_url.format(index=index)
    output_path = os.path.join(output_dir, f"{index}.parquet")
    
    try:
        print(f"Downloading {file_url}...")
        response = requests.get(file_url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Write file to disk
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                
        print(f"Saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_url}: {e}")

print("Download completed!")
