import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from core.utils import scandir
import tifffile


def main():
    """Tool to crop large images to sub-images for faster IO.

    It is used for SN6 dataset.

    opt (dict): Configuration dict. It contains:
        input_folder (str):  Path to dir with large images to split on regions
        save_folder (str): Path to dir to save images regions
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower
            than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are 2 folders to be processed for SN6 dataset.
            SN6_train_HR
            SN6_train_LR_bicubic
        After process, each sub_folder should have the same number of
        subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}

    # HR images
    opt['input_folder'] = 'datasets/SN6/expanded/PS-RGBNIR'
    opt['save_folder'] = 'datasets/SN6/splitted/PS-RGBNIR'
    opt['crop_size'] = 512
    opt['step'] = 256
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # LRx4 images
    opt['input_folder'] = 'datasets/SN6/expanded/RGBNIR'
    opt['save_folder'] = 'datasets/SN6/splitted/RGBNIR'
    opt['crop_size'] = 128
    opt['step'] = 64
    opt['thresh_size'] = 0
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')

    img_list = list(scandir(input_folder, full_path=True))

    starting_name_index = 0
    for path in img_list:
        starting_name_index = split(path, opt, starting_name_index)

    print('All processes done.')


def split(path: str, opt, index: int = 0):
    """Splits image on regions

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
        index (int): starting name for subimages

    Return:
        index (name) of the last saved image, so next image index can start from (index + 1) name
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename
    img_name = img_name.replace('x2', '').replace(
        'x3', '').replace('x4', '').replace('x8', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, c = img.shape
    else:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')

    if c > h or c > w:
        raise ValueError(f'Wrong image shape: {img.shape}. Should be HWC')

    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    prog = tqdm(total=len(h_space)*len(w_space))
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imsave(osp.join(opt['save_folder'],
                                f'{index:04d}{extension}'), cropped_img)
            prog.update()

    return index


if __name__ == '__main__':
    main()
