import cv2
import os
import numpy as np
import shutil
import math
from enum import Enum
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# --------------------
# Filesystem functions
# --------------------


def create_folder(folder: str):
    """ Creates folder if it dosn't exist yet """
    if not os.path.exists(folder):
        os.makedirs(folder)


def clear_folder_content(folder):
    """ Deletes all files in the directory """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def copy_images(image_name_list: list, src_dir: str, dst_dir: str):
    """ Copies all files from src directory to dst directory """
    for image_name in tqdm(image_name_list):
        image_path = os.path.join(src_dir, image_name)
        shutil.copy(image_path, dst_dir)

# ----------------------
# Image transformations
# ----------------------


class NoiseType(Enum):
    Gauss = 1
    SandP = 2
    Poisson = 3
    Speckle = 4


def noisy(image: np.array, noise_type: NoiseType):
    """ Adds noise to an image """

    if noise_type == NoiseType.Gauss:
        row, col, ch = image.shape

        mean = 0
        sigma = 0.01
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_type == NoiseType.SandP:
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_type == NoiseType.Poisson:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_type == NoiseType.Speckle:
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch) * 0.01
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

    else:
        return image


def transform_single_image(image_name: str, src_dir: str, dst_dir: str, image_height: int, image_width: int, noises: list) -> bool:
    """ Loads image from path, resizes it to a [image_height x image_width] size and adds noise """

    try:
        # read image
        image_path = os.path.join(src_dir, image_name)
        image = cv2.imread(image_path)
        image_max = np.max(image)
        image_min = np.min(image)

        # transform
        image = cv2.resize(image, (image_height, image_width),
                           interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float)
        cv2.normalize(image, image, alpha=0, beta=1,
                      norm_type=cv2.NORM_MINMAX, dtype=-1)

        for noise in noises:
            image = noisy(image, noise)

        # convert to uint8 image
        cv2.normalize(image, image, alpha=image_max, beta=image_min,
                      norm_type=cv2.NORM_MINMAX, dtype=-1)
        image = image.astype(np.uint8)

        # save
        dst_path = os.path.join(dst_dir, image_name)
        cv2.imwrite(dst_path, image)

    except Exception as e:
        print(str(e))
        return False
    return True


def transform_images(images_name_list: list, src_dir: str, dst_dir: str, image_height: int, image_width: int, noises: list):
    """ Resizes all images in the src directory by a scale factor and adds specified noise
        and saves them to dst directory """

    num_cores = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(total=len(images_name_list)) as progress:
            futures = []

            for image_name in images_name_list:
                future = pool.submit(transform_single_image,
                                     image_name, src_dir, dst_dir, image_height, image_width, noises)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result)


def prepare_dataset(images_dir: str, split_rate: float,
                    hr_noises: list, lr_noises: list,
                    hr_height: int, hr_width: int,
                    lr_height: int, lr_width: int,
                    out_dataset_dirname: str):

    hr_train_dir = os.path.join(out_dataset_dirname, "train/high_res")
    lr_train_dir = os.path.join(out_dataset_dirname, "train/low_res")
    hr_val_dir = os.path.join(out_dataset_dirname,   "val/high_res")
    lr_val_dir = os.path.join(out_dataset_dirname,   "val/low_res")

    images_names = os.listdir(images_dir)
    images_count = len(images_names)
    train_image_names = images_names[:int(images_count * split_rate)]
    val_image_names = images_names[int(images_count * split_rate):]

    print("number of images for training:", len(train_image_names))
    print("number of images for validation:", len(val_image_names))

    # create empty folders
    for folder in [hr_train_dir, hr_val_dir, lr_train_dir, lr_val_dir]:
        create_folder(folder)
        clear_folder_content(folder)

    # high resolution images
    print("Prepare training set of high resolution images in", hr_train_dir, "\n")
    transform_images(train_image_names,
                     src_dir=images_dir,
                     dst_dir=hr_train_dir,
                     image_height=hr_height,
                     image_width=hr_width,
                     noises=hr_noises)
    print("Prepare validation set of high resolution images in", hr_val_dir, "\n")
    transform_images(val_image_names,
                     src_dir=images_dir,
                     dst_dir=hr_val_dir,
                     image_height=hr_height,
                     image_width=hr_width,
                     noises=hr_noises)

    # low resolution images
    print("Prepare training set of low resolution images in", lr_train_dir, "\n")
    transform_images(train_image_names,
                     src_dir=images_dir,
                     dst_dir=lr_train_dir,
                     image_height=lr_height,
                     image_width=lr_width,
                     noises=lr_noises)
    print("Prepare validation set of low resolution images in", lr_val_dir, "\n")
    transform_images(val_image_names,
                     src_dir=images_dir,
                     dst_dir=lr_val_dir,
                     image_height=lr_height,
                     image_width=lr_width,
                     noises=lr_noises)


if __name__ == '__main__':
    images_dir = "datasets/Argis/images"
    split_rate = 0.95  # training - validation split rate
    hr_noises = []  # No noise for high resolution images
    # Noise for low resolution images
    lr_noises = [NoiseType.Gauss, NoiseType.Speckle]
    hr_height, hr_width = (336, 336)  # new size for high res images
    lr_height, lr_width = (84, 84)  # new size for low res images
    out_dataset_dirname = "datasets/Argis/dataset_30sm_speckle_001_noise"

    prepare_dataset(images_dir, split_rate,
                    hr_noises, lr_noises,
                    hr_height, hr_width,
                    lr_height, lr_width,
                    out_dataset_dirname)
