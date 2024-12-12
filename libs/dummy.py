import random
from pathlib import Path

from PIL import Image


dummy_images = [file for file in Path("./dummy_scans").glob("*.png")]

def get_random_dummy_image_path():
    return dummy_images[random.randrange(len(dummy_images))]

def get_random_dummy_image():
    file = get_random_dummy_image_path()
    print(f"Opening image file '{file}'...")
    return Image.open(file)