import numpy as np
from Registration_src.Utils.Visualization import save_comparison_fig
from Registration_src.Landmark_Matching import Landmark_Description_Set
import cv2

"""
Functions for performing registration between FA-FA
"""

def Register_FA_Sequence(FA_img_sequence, base_img_no, path):
    """
    Register sequence of FA images
    :param FA_img_sequence: list of FA images
    :param base_img_no: index of FA image list for initial base/reference image (frame which other frames will be
     registered towards)
    :param path: preview save path
    :return: registered_images: list of registered images
    """
    ref_img = FA_img_sequence[base_img_no]
    registered_images1 = []

    # Perform two passes of registration (increases accuracy)

    print("First registration pass")
    for i, target_img in enumerate(FA_img_sequence):
        print("First Pass: Registering image {} of {}".format(i+1, len(FA_img_sequence)))
        # Calculate homography/transformation matrix
        homography = FA_FA_registration_transform(refImg_color=ref_img,
                                                  targetImg_color=target_img,
                                                  path=path)
        # Transform image using homography
        transformed_img = cv2.warpPerspective(np.array(target_img), homography,
                                              (target_img.shape[1], target_img.shape[0]))
        ref_img = np.copy(transformed_img)

        # Add transformed image to list
        registered_images1.append(transformed_img)

    # Seconds registration pass
    print("Second registration pass")
    ref_img = registered_images1[base_img_no]
    registered_images2 = []
    for i, target_img in enumerate(registered_images1):
        print("Second Pass: Registering image {} of {}".format(i+1, len(FA_img_sequence)))
        homography = FA_FA_registration_transform(refImg_color=ref_img,
                                                  targetImg_color=target_img,
                                                  path=path)
        transformed_img = cv2.warpPerspective(np.array(target_img), homography,
                                              (target_img.shape[1], target_img.shape[0]))
        ref_img = np.copy(transformed_img)
        registered_images2.append(transformed_img)

    return registered_images2


def FA_FA_registration_transform(refImg_color, targetImg_color, path=None):
    '''
    Performs registration between two FA images. Uses both ORB and SIFT landmarks where the optic disk area is masked
    out in the former.
    :param refImg_color: reference image numpy array
    :param targetImg_color: target image to be transformed. Also a numpy array
    :param path: path for saving debugging/preview images. 'None' as default
    :return: homography: homography matrix
    '''
    target_scale = targetImg_color.shape

    # Normalize images
    img_target = np.array(targetImg_color)
    img_target = (img_target / np.max(img_target) * 255.0).astype('uint8')
    img_ref = np.array(refImg_color)
    img_ref = (img_ref / np.max(img_ref) * 255.0).astype('uint8')

    if len(img_target.shape) == 3:
        img_target = cv2.cvtColor(np.array(img_target), cv2.COLOR_BGR2GRAY)
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(np.array(img_ref), cv2.COLOR_BGR2GRAY)

    # Invert images (white background tends to work better)
    img_target = 255 - img_target
    img_ref = 255 - img_ref

    # Preprocess image with adaptive histogram normalization
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))
    img_target = clahe.apply(img_target)
    img_ref = clahe.apply(img_ref)


    if path is not None:
        save_comparison_fig(img_ref, img_target, img_ref.shape[0], path + 'normalized.png')

    # Detect set of landmarks using combined ORB, AKAZE, HOG, SIFT, ect.
    # (Can instead use GPU-accelerated ORB detection)
    p1, p2 = Landmark_Description_Set(img_ref=img_ref, img_target=img_target, path=path)

    homography, mask = cv2.findHomography(p2, p1, cv2.RANSAC)

    return homography
