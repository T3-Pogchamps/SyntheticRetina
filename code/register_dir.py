from os import listdir
import cv2
import os.path

from numpy import source
from Registration import FA_FA_registration_transform, Register_FA_Sequence

if __name__ == "__main__":
    
    target_dir = "../data/registered_sample_seq/"
    source_dir = "../data/sample_seq/"
    subdirs = listdir(source_dir)
    
    print(source_dir, target_dir, subdirs)
    for i, subdir in enumerate(subdirs[0:]):
        print("Processing subdirectory {} of {}".format(i+1, len(subdirs)))
        image_names_list = listdir(source_dir+subdir)
        print(image_names_list)
        source_imgs = []
        base_img_no = -1
        for counter, f in enumerate(image_names_list):
            source_imgs.append(cv2.imread(source_dir+subdir+"/"+f))
        
        registered_FA_sequence = Register_FA_Sequence(FA_img_sequence=source_imgs, base_img_no=0, path=(target_dir+subdir+"/"))
        for i, transformed_img in enumerate(registered_FA_sequence):
            cv2.imwrite(target_dir+subdir+"/"+"{}.png".format(i), transformed_img)
