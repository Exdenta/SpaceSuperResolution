
import os
import cv2
import numpy as np
from PIL import Image
from tifffile import tifffile
import imquality.brisque as brisque
from skimage import io

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

images_dir = "Images_Openaerialmap"
image_names = os.listdir(images_dir)


def load_img(filepath):
    image = tifffile.imread(filepath)  # open tiff file in read mode
    return image


for image_name in image_names:
    image_path = os.path.join(images_dir, image_name)

    if ((os.path.getsize(image_path) // (1024 * 1024)) < 500):

        # Load image
        image = load_img(image_path)

        # crop center
        center_h, center_w, _ = image.shape
        center_h = center_h // 2
        center_w = center_w // 2
        image = image[center_h:center_h + 1024, center_w:center_w + 1024, :]

        score = brisque.score(image)
        print("image:", image_path, "\tscore:", score)
