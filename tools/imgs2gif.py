import os
import re
from PIL import Image

def natural_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


def folder_to_gif(folder, output_file, duration=500, loop=0):
    all_images = []

    images = sorted(
        [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: natural_key(os.path.basename(x))  # Apply natural sorting based on the base filename
    )

    for img_path in images:
        if "text_guide_edit.png" in img_path:
            try:
                img = Image.open(img_path).convert("RGBA")  # Convert to RGBA for consistency
                all_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    if not all_images:
        print("No images found to create a GIF.")
        return

    # Save as GIF
    all_images[0].save(
        output_file,
        save_all=True,
        append_images=all_images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    print(f"GIF saved as {output_file}")

# Example usage
folder = "/lustre/scratch/client/scratch/research/group/khoigroup/tungnt132/os-personalize/debug_vis"  # Replace with your folder paths
output_file = "../output.gif"
folder_to_gif(folder, output_file, duration=200)