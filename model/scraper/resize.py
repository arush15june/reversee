import os
from PIL import Image

IMAGES_DIR = 'images'
OUTPUT_DIR = os.path.join(IMAGES_DIR, 'resized')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

ALL_IMAGES = os.listdir(IMAGES_DIR)

def resizeImage(image):
    return image.resize((180, 240), Image.ANTIALIAS).crop((20, 50, 150, 200))

for file_name in ALL_IMAGES:
    if 'jpg' not in file_name:
        continue

    image = Image.open(os.path.join(IMAGES_DIR, file_name))
    image_crop = resizeImage(image)
    image_crop.save(os.path.join(OUTPUT_DIR, file_name))
    print(f"Cropped {file_name}")