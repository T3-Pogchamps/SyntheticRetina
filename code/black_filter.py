import os.path
import shutil

import numpy as np
from PIL import Image


def Is_Black_Image(image_PIL, unique_thresh=150, max_pixel_thresh=120, show_img=False):
    """
    Return "True" when a black/noisy image is detected
    :param image_PIL: image data (loaded with PIL.image)
    :param unique_thresh: minimum number of unique pixel values to be considers not 'blank' (0->256)
    :param max_pixel_thresh: minimum maximum pixel value to be considered not 'blank' (0->256)
    :param show_img: option to show preview image and calculated parameters
    :return: 'True' for black image, 'False' otherwise
    """
    data = np.asarray(image_PIL)
    np.reshape(data, (-1, 1))
    u, count_unique = np.unique(data, return_counts=True)
    no_unique = len(count_unique)
    std = np.std(data / np.max(data))
    max_pix = np.max(data)

    is_black = False

    if max_pix < max_pixel_thresh:
        is_black = True
    else:
        unique_thresh -= 15
    if no_unique < unique_thresh:
        is_black = True

    if show_img:
        plt.imshow(image_PIL, cmap='gray')
        plt.axis('off')
        plt.title("unique: {}, std: {}, max: {} => IS_BLACK: {}".format(no_unique, std, max_pix, is_black))
        plt.show()

    return is_black


def Remove_Blank_Images(source_dir, DETECT_ONLY=False):
    """
    Scan images in a directory and identify blank images. (optional) delete blank images
    :param source_dir:
    :param DETECT_ONLY:
    :return:
    """

    files = listdir(source_dir)
    no_black_imgs = 0
    for f in files:
        image = Image.open(source_dir + f).convert('L')
        is_black_img = Is_Black_Image(image, show_img=False)
        if is_black_img:
            if DETECT_ONLY:
                print("{} is black image".format(f))
            else:
                remove(source_dir + f)
                no_black_imgs += 1
    print("removed {} black images".format(no_black_imgs))


def Remove_Blank_Images_Subdir(source_dir, DETECT_ONLY=False, save_blank_img_path=None):
    """
    Scan images within subdirectories in a specified directory and identify blank images. (optional) delete blank images
    :param source_dir:
    :param DETECT_ONLY:
    :return:
    """

    no_black_imgs = 0
    subdirs = listdir(source_dir)
    for i, subdir in enumerate(subdirs[90:]):
        print("Processing subdirectory {} of {}".format(i+1, len(subdirs)))
        files = listdir(source_dir+subdir)
        for f in files:
            try:
                image = Image.open(source_dir + subdir + "/" + f).convert('L')
                is_black_img = Is_Black_Image(image, show_img=False, unique_thresh=100)
                if is_black_img:
                    # if save_blank_img_path is specified, move the blank image to the specified directory
                    if save_blank_img_path is not None:
                        if not os.path.exists(save_blank_img_path + "/" + subdir):
                            os.mkdir(save_blank_img_path + "/" + subdir)
                        shutil.move(src=source_dir + subdir + "/" + f, dst=save_blank_img_path + "/" + subdir + "/" + f)
                    # if "DETECT_ONLY", don't remove the files, only print the detection message
                    elif DETECT_ONLY:
                        print("{} is black image".format(f))
                    # else, delete the image
                    else:
                        remove(source_dir + subdir + "/" + f)
                        no_black_imgs += 1
            except:
                print("FAILED TO OPEN IMAGE: {}".format(source_dir + subdir + "/" + f))
    print("removed {} black images".format(no_black_imgs))


if __name__ == "__main__":
    from os import listdir, remove
    from os.path import join, isfile
    import matplotlib.pyplot as plt

    #Remove_Blank_Images(source_dir="Z:/datasets/Retina/")
    Remove_Blank_Images_Subdir(source_dir="Y:/datasets/Retina/",
                               save_blank_img_path="Y:/datasets/Retina/")

    #res = Is_Black_Image(image_PIL=Image.open("Y:/datasets/Retina/FluoresceinAngiography/CAPSTONE FA Sequences/blank/460R-21_15_05_51_559.png").convert('L'),
    #                     unique_thresh=100)
    #print(res)