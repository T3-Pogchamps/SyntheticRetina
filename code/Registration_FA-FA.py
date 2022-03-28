from os import listdir
import cv2
from Registration import FA_FA_registration_transform, Register_FA_Sequence

"""
Performed registration on a specified sequence of FAs using functions from Registration_src.py
"""

if __name__ == "__main__":

    # Path to save registered images
    OUTPUT_DIR = "../data/registered_sample_seq/seq10/"

    # Path to sequence of images to register
    target_sequence_dir = '../data/sample_seq/seq10/'

    # Load image sequence
    image_names_list = listdir(target_sequence_dir)
    print(image_names_list)
    target_imgs = []
    base_img_no = -1
    for counter, f in enumerate(image_names_list):
        target_imgs.append(cv2.imread(target_sequence_dir + f))
    print(target_imgs)

    # Register images to frame number 'base_img_no'
    # Save preview images tot 'path'
    registered_FA_sequence = Register_FA_Sequence(FA_img_sequence=target_imgs, base_img_no=0, path=OUTPUT_DIR)

    # Save registered sequence
    for i, transformed_img in enumerate(registered_FA_sequence):
        cv2.imwrite(OUTPUT_DIR + "{}.png".format(i), transformed_img)

