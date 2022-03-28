from os import getcwd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
try:
    from .Preprocessing.Preprocess import Segment_Veins_SLO, Segment_Veins_FA
    from .Utils.Visualization import CompositeChannelImages, Reframe_Image, Crop_Image_Window
except:
    sys.path.append(getcwd().split('Image_Registration_openCV')[0] + 'Image_Registration_openCV')
    from Preprocessing.Preprocess import Segment_Veins_SLO, Segment_Veins_FA
    from Utils.Visualization import CompositeChannelImages, Reframe_Image, Crop_Image_Window

from scipy import signal

"""
Functions for performing initial alignment of FA and SLO images using correlation, convolution and gradient-based
approaches.
"""


def Alignment_Parameters_to_Homography(target_img_size, scale, offset):
    """
    Generate a 3x3 homography matrix from the calculate alignment scale and offset
    :param target_img_size: shape of original target image (typically SLO for SLO->FA registration)
    :type target_img_size: numpy array [2]
    :param scale: alignment scale factor (X,Y). note: this is the scaling factor, not the width/height
    :type scale: numpy array [2]
    :param offset: (X,Y) alignment offset (center coordinate of aligned image) in pixels
    :type offset: numpy array [2]
    :return: Homography matrix
    :rtype: numpy array [3,3]
    """
    # calculate scale in pixels
    new_scale = np.multiply((target_img_size[0], target_img_size[1]), (scale[1], scale[0]))
    # bottom left corner coordinates before and after transform
    bot_left_corner = [(0, 0), (offset[0] - new_scale[0] / 2, offset[1] - new_scale[1] / 2)]
    # top right corner coordinates before and after transform
    top_right_corner = [(target_img_size[0], target_img_size[1]),
                        (offset[0] + new_scale[0] / 2, offset[1] + new_scale[1] / 2)]

    # generate source destination points list
    src_pts = []
    dst_pts = []
    src_pts.append((bot_left_corner[0][0], bot_left_corner[0][1]))  # bottom left
    src_pts.append((top_right_corner[0][0], top_right_corner[0][1]))  # top right
    src_pts.append((bot_left_corner[0][0], top_right_corner[0][1]))  # top left
    src_pts.append((top_right_corner[0][0], bot_left_corner[0][1]))  # bottom right

    dst_pts.append((bot_left_corner[1][0], bot_left_corner[1][1]))  # bottom left
    dst_pts.append((top_right_corner[1][0], top_right_corner[1][1]))  # top right
    dst_pts.append((bot_left_corner[1][0], top_right_corner[1][1]))  # top left
    dst_pts.append((top_right_corner[1][0], bot_left_corner[1][1]))  # bottom right

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    # calculate homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts)
    return H


def Crop_Image_Homography(window_size, window_center):
    """
    Crop an image to a specified window size and center
    :param window_size: (W2,H2) window size in pixels
    :type window_size: numpy array [2]
    :param window_center: (X,Y) coordinate of window center
    :type window_center: numpy array [2]
    :return: cropped image
    :rtype: numpy array [W2,H2]
    """

    img_start_idx = [int(window_center[0] - window_size[1] / 2),
                     int(window_center[1] - window_size[0] / 2)]
    img_end_idx = [int(window_center[0] + window_size[1] / 2),
                   int(window_center[1] + window_size[0] / 2)]

    # bottom left corner coordinates before and after transform
    bot_left_corner = [(img_start_idx[0], img_start_idx[1]), (0, 0)]
    # top right corner coordinates before and after transform
    top_right_corner = [(img_end_idx[0], img_end_idx[1]), (window_size[1], window_size[0])]

    # generate source destination points list
    src_pts = []
    dst_pts = []
    src_pts.append((bot_left_corner[0][0], bot_left_corner[0][1]))  # bottom left
    src_pts.append((top_right_corner[0][0], top_right_corner[0][1]))  # top right
    src_pts.append((bot_left_corner[0][0], top_right_corner[0][1]))  # top left
    src_pts.append((top_right_corner[0][0], bot_left_corner[0][1]))  # bottom right

    dst_pts.append((bot_left_corner[1][0], bot_left_corner[1][1]))  # bottom left
    dst_pts.append((top_right_corner[1][0], top_right_corner[1][1]))  # top right
    dst_pts.append((bot_left_corner[1][0], top_right_corner[1][1]))  # top left
    dst_pts.append((top_right_corner[1][0], bot_left_corner[1][1]))  # bottom right

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    # calculate homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts)
    return H


def Position_Correlation_Score(img_1, img_2, offset_sweep_range=100, offset_sweep_steps=8, start_pos=None):
    """
    Sweep img_2 across img_1, calculating the correlation at each offset step. Find the 2D gradient of the correlations
    at the maximum score offset location to calculate an overall correlation score at the optimal offset.
    :param img_1: base image
    :type img_1: numpy array [W1, H1]
    :param img_2: image to be stepped across img_1
    :type img_2: numpy array [W2, H2]
    :param offset_sweep_range: range (in pixels) of alignment offset sweep (default: 100)
    :type offset_sweep_range: int
    :param offset_sweep_steps: number of alignment offset sweep steps (default: 8)
    :type offset_sweep_steps: int
    :param start_pos: (X,Y) initial position guess (default=None, use center of reference image).
    :type start_pos: numpy array [2]
    :return: score: correlation score (float)
        best_offset: (x,y) position of best offset from the start position
    :rtype:
    """
    if start_pos is None:
        start_pos = (img_1.shape[1] / 2, img_1.shape[0] / 2)
    if offset_sweep_range == 0:
        offset_sweep_range = 0.001
    x_sweep = range(-offset_sweep_range, offset_sweep_range, offset_sweep_steps)  # range of x-offsets to try
    y_sweep = range(-offset_sweep_range, offset_sweep_range, offset_sweep_steps)  # range of y-offsets to try

    conv = np.zeros((len(x_sweep), len(y_sweep)))  # matrix for storing convolution values
    for ix in range(len(x_sweep)):
        for iy in range(len(y_sweep)):
            # Crop img_1 down to the same window as the where img_b would be after scaling and offsetting
            offset = (start_pos[0] + x_sweep[ix], start_pos[1] + y_sweep[iy])
            img_1_cropped = Crop_Image_Window(img=img_1, window_size=img_2.shape, window_center=offset)
            # multiply the cropped img_1 with scaled img_b and then sum the result to obtain the correlation
            overlap = np.sum(np.multiply(img_1_cropped, img_2))
            # calculate score by normalizing overlap with scaled img_b size
            conv[ix, iy] = overlap / np.power(img_2.shape[0] * img_2.shape[1], 0.7)

    # find maximum convolution value across all candidate offsets for this scaling factor
    offset_xm, offset_ym = np.unravel_index(conv.argmax(), conv.shape)
    best_offset = (start_pos[0] + x_sweep[offset_xm], start_pos[1] + y_sweep[offset_ym])
    # calculate the 2D spatial gradient
    gx, gy = np.gradient(conv)
    G = np.abs(gx) + np.abs(gy)
    # Calculate the gradient value (averaged around the max convolution pixel)
    G_score_multiplier = np.sum(G[(offset_xm - 1):(offset_xm + 1), (offset_ym - 1):(offset_ym + 1)])
    # score value for this scaling factor weighted by the gradient magnitude
    score = np.max(conv) * G_score_multiplier
    return score, best_offset


def Initial_Align_Images(img_FA_vein, img_SLO_vein, subscaling_factor=2,
                         scale_sweep_range=0.20, scale_sweep_steps=12,
                         scale_AR_sweep_range=0.25, scale_AR_sweep_steps=12,
                         offset_sweep_range=100, offset_sweep_steps=18, start_scale=0.3,
                         start_pos=None, threads=0):
    """
    Determing optimal alignment scale and offset for overlapping SLO onto FA based on vein overlap correlation.
    :param img_FA_vein: binary image containing segmented veins from FA image (larger image)
    :type img_FA_vein: numpy array [W1, H1]
    :param img_SLO_vein: binary image containing segmented veins from SLO image (smaller image)
    :type img_SLO_vein: numpy array [W2, H2]
    :param subscaling_factor: factor to downscale both image arrays for faster processing.
     Higher number = more downscaling
    :type subscaling_factor: integer
    :param scale_sweep_range: range of scale multipliers to try (default: 0.2)
    :type scale_sweep_range: float
    :param scale_sweep_steps: number of scale steps (default: 10)
    :type scale_sweep_steps: int
    :param scale_AR_sweep_range: range of scale aspect ratio multipliers to try (default: 0.2)
    :type scale_AR_sweep_range: float
    :param scale_AR_sweep_steps: number of scale aspect ratio steps (default: 5)
    :type scale_AR_sweep_steps: int
    :param offset_sweep_range: range (in pixels) of alignment offset sweep (default: 100)
    :type offset_sweep_range: int
    :param offset_sweep_steps: number of alignment offset sweep steps (default: 8)
    :type offset_sweep_steps: int
    :param start_scale: initial scale guess (default=0.5) assumed to be square, relative to original image scale
    :type start_scale: float
    :param start_pos: (X,Y) initial position guess (default=None).
     If 'None', use the center of the img_FA frame as initial
    :type start_pos: numpy array [2]
    :return: alignment_scale: scaling factor (0->1) for SLO alignment
        alignment_offset: center coordinates (in pixels) for aligning scaled SLO onto FA frame
        best_alignment_score: correlation score for alignment output
    :rtype:
    """

    # Apply downscaling
    img_a = cv2.resize(img_FA_vein, (int(img_FA_vein.shape[1] / subscaling_factor),
                                     int(img_FA_vein.shape[0] / subscaling_factor)))
    img_b = cv2.resize(img_SLO_vein, (int(img_SLO_vein.shape[1] / subscaling_factor),
                                      int(img_SLO_vein.shape[0] / subscaling_factor)))
    img_a = img_a / np.max(img_a)
    img_b = img_b / np.max(img_b)

    if start_pos is None:
        start_pos = (img_a.shape[1] / 2, img_a.shape[0] / 2)

    ## Manual scale sweep:
    scale_score = []
    scale_track = []
    pos_track = []
    scale_sweep = start_scale + np.linspace(-scale_sweep_range, scale_sweep_range,
                                            scale_sweep_steps)  # range of scales to try

    # iterate over possible square scale values

    # If enabled, use multithreading
    if threads > 0:
        import threading

        def worker(scales, results_temp):
            for i in range(len(scales)):
                img_b_scale = scales[i]  # candidate scale

                # generate scaled img_b
                img_b_scaled = cv2.resize(img_b,
                                          (int(img_b.shape[1] * img_b_scale[1]), int(img_b.shape[0] * img_b_scale[0])))

                # Sweep offset to find the correlation score and highest scoring offset
                corr_score, best_offset = Position_Correlation_Score(img_1=img_a, img_2=img_b_scaled,
                                                                     offset_sweep_range=offset_sweep_range,
                                                                     offset_sweep_steps=offset_sweep_steps,
                                                                     start_pos=start_pos)

                # store the score value for this scaling factor weighted by the gradient magnitude
                # Larger gradient = faster dropoff of overlap when shifting img_b = more likely that this is a good vein match
                # scale_score.append(corr_score)
                # scale_track.append(img_b_scale)
                # pos_track.append(best_offset)
                results_temp.append([corr_score, img_b_scale, best_offset])

        t = []
        t_res = []
        test_scales = []
        for i_scale in range(len(scale_sweep)):
            scale_ar_sweep = np.linspace(scale_sweep[i_scale] * (1 - scale_AR_sweep_range),
                                         scale_sweep[i_scale] * (1 + scale_AR_sweep_range),
                                         scale_AR_sweep_steps)
            for i_ar in range(len(scale_ar_sweep)):
                test_scales.append((scale_sweep[i_scale], scale_ar_sweep[i_ar]))
        test_scales_split = np.array_split(test_scales, threads)
        for i in range(threads):
            t_res.append([])
            t_new = threading.Thread(target=worker, args=(test_scales_split[i], t_res[i]))
            t_new.start()
            t.append(t_new)
        for n, thread in enumerate(t):
            thread.join()
        for i in range(threads):
            for j in range(len(t_res[i])):
                scale_score.append(t_res[i][j][0])
                scale_track.append(t_res[i][j][1])
                pos_track.append(t_res[i][j][2])
    else:
        for i_scale in range(len(scale_sweep)):
            scale_ar_sweep = np.linspace(scale_sweep[i_scale] * (1 - scale_AR_sweep_range),
                                         scale_sweep[i_scale] * (1 + scale_AR_sweep_range),
                                         scale_AR_sweep_steps)

            for i_ar in range(len(scale_ar_sweep)):
                img_b_scale = (scale_sweep[i_scale], scale_ar_sweep[i_ar])  # candidate scale

                # generate scaled img_b
                img_b_scaled = cv2.resize(img_b,
                                          (int(img_b.shape[1] * img_b_scale[1]), int(img_b.shape[0] * img_b_scale[0])))
                # Sweep offset to find the correlation score and highest scoring offset
                corr_score, best_offset = Position_Correlation_Score(img_1=img_a, img_2=img_b_scaled,
                                                                     offset_sweep_range=offset_sweep_range,
                                                                     offset_sweep_steps=offset_sweep_steps,
                                                                     start_pos=start_pos)
                # store the score value for this scaling factor weighted by the gradient magnitude
                # Larger gradient = faster dropoff of overlap when shifting img_b = more likely that this is a good vein match
                scale_score.append(corr_score)
                scale_track.append(img_b_scale)
                pos_track.append(best_offset)
                '''
                img_b_framed, img_b_scaled = Reframe_Image(img=img_a, new_scale=img_b_scale, new_offset=best_offset,
                                                           frame_size=img_a.shape)
                plt.imshow(img_a, cmap='gray')
                plt.imshow(img_b_framed, cmap='hot', alpha=0.5)
                plt.title("score: {:.2f} (best={:.2f}), scale: {}".format(corr_score, np.array(scale_score).max(), img_b_scale))
                plt.show()
                '''
        # print("[Initial_Align_Images] Scale {}: {},  Best Score: {}".format(i_scale, scale_sweep[i_scale], np.max(conv)))

    # Find best alignment scale and offset
    nmax = np.argmax(np.array(scale_score))
    best_scale = scale_track[nmax]
    best_offset = pos_track[nmax]
    print("[Initial_Align_Images] Best alignment (score={}): scale={}, offset={}".format(
        np.max(np.array(scale_score)), best_scale, best_offset))

    return best_scale, np.array(best_offset) * subscaling_factor, np.max(np.array(scale_score)) * subscaling_factor


if __name__ == "__main__":
    import time
    from DL_Vein_Segmentation.FA_Vein_Segmentation.Evaluate import DL_FA_Segment_Veins
    from DL_Vein_Segmentation.SLO_Vein_Segmentation.Evaluate import DL_SLO_Segment_Veins

    IMG_DIR = getcwd().split('\\Registration_src')[0] + "/Images/"
    OUTPUT_DIR = getcwd().split('\\Registration_src')[0] + "/Output/CorrelationMatching/"

    ref_img_path = IMG_DIR + "FA12.png"
    target_img_path = IMG_DIR + "SLO12.png"

    #ref_img_path = "E:/Datasets/FA_OCT_registration/ValidationPairs/3/FA.png"
    #target_img_path = "E:/Datasets/FA_OCT_registration/ValidationPairs/3/SLO.png"

    img_FA = cv2.imread(ref_img_path)
    img_SLO = cv2.imread(target_img_path)

    img_SLO = np.array(img_SLO)
    img_SLO = (img_SLO / np.max(img_SLO) * 255.0).astype('uint8')
    img_FA = np.array(img_FA)
    img_FA = (img_FA / np.max(img_FA) * 255.0).astype('uint8')

    use_ml_vein_seg = True
    if not use_ml_vein_seg:
        img_FA_vein = Segment_Veins_FA(img_FA=img_FA, small_obj_filter=240)
        img_SLO_vein = Segment_Veins_SLO(img_SLO=img_SLO)
    else:
        img_FA_vein = DL_FA_Segment_Veins(fa=img_FA, threshold=25, small_obj_filter=30, USE_THRESH=True)
        img_SLO_vein = DL_SLO_Segment_Veins(slo=img_SLO, threshold=50, small_obj_filter=300, USE_THRESH=True)

    cv2.imwrite(OUTPUT_DIR + "FA_veins.png", img_FA_vein)
    cv2.imwrite(OUTPUT_DIR + "SLO_veins.png", img_SLO_vein)

    t0 = time.time()
    align_scale, align_pos, _ = Initial_Align_Images(img_FA_vein=img_FA_vein, img_SLO_vein=img_SLO_vein,
                                                     subscaling_factor=2, threads=0)
    print("alignment time: {}".format(time.time() - t0))

    print("Alignment scale: {}, offset: {}".format(align_scale, align_pos))
    img_b_framed, img_b_scaled = Reframe_Image(img=img_SLO_vein, new_scale=align_scale, new_offset=align_pos,
                                               frame_size=img_FA_vein.shape)

    H = Alignment_Parameters_to_Homography(target_img_size=img_SLO.shape, scale=align_scale, offset=align_pos)

    im_out = cv2.warpPerspective(img_SLO, H, (img_FA.shape[1], img_FA.shape[0]))

    comp = CompositeChannelImages(img_FA_vein, img_b_framed, b_intensity=1)
    # plt.cla()
    plt.figure(figsize=(20, 20))
    plt.imshow(comp)
    plt.savefig(OUTPUT_DIR + "aligned_images.png")
    # plt.show()

