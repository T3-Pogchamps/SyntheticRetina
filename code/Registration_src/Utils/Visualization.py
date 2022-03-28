from os import getcwd
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
Code for visualizing registration results
"""

def ExtendToRGBchannels(img):
    """
    Convert an (image data) array from shape [W,H] to [3,W,H]
    :param img: input image
    :return: output image
    """
    img_3ch = np.zeros((3, img.shape[0], img.shape[1]))
    for i in range(0, 3):
        img_3ch[i, :, :] = img
    img_3ch = (img_3ch / np.max(img_3ch) * 255.0).astype('uint8')
    return img_3ch


def CompositeOverlayImages(img_back, img_front):
    """
    Composite two images based on alpha transparency
    :param img_back: background image
    :type img_back: numpy array [W,H,3] or [W,H,4]
    :param img_front: foreground image
    :type img_front: numpy array [W,H,3] or [W,H,4]
    :return: composite image
    :rtype: numpy array [W,H,4]
    """
    composite = np.zeros(img_back.shape)

    if img_back.shape[2] == 3:
        img_back = cv2.cvtColor(img_back, cv2.COLOR_RGB2RGBA)
    if img_front.shape[2] == 3:
        img_front = cv2.cvtColor(img_front, cv2.COLOR_RGB2RGBA)

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = img_back[:, :, 3] / 255.0
    alpha_foreground = img_back[:, :, 3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        composite[:, :, color] = alpha_foreground * img_front[:, :, color] + \
                                 alpha_background * img_back[:, :, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    composite[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return composite.astype('uint8')


def CompositeChannelImages(img_a, img_b, b_intensity=0.5):
    """
    Composite two images base on color channels. Images must be the same size.
    :param img_a: Background image (red channel)
    :type img_a: Numpy array [W,H,3]
    :param img_b: Foreground image (green/blue image)
    :type img_b: Numpy array [W,H,3]
    :param b_intensity: Color intensity of img b
    :type b_intensity: float 0->1
    :return: composite image
    :rtype:Numpy array [W,H,3]
    """
    img_composite = np.zeros((img_a.shape[0], img_a.shape[1], 3))
    if len(img_a.shape) == 3:
        img_a = np.mean(np.array(img_a), 2)
    if len(img_b.shape) == 3:
        img_b = np.mean(np.array(img_b), 2)
    img_composite[:, :, 0] = img_a / np.max(img_a)
    img_composite[:, :, 1] = b_intensity * (img_b / np.max(img_b))
    img_composite[:, :, 2] = b_intensity * (img_b / np.max(img_b))
    img_composite = (img_composite * 255.0).astype('uint8')

    return img_composite


def Preview_Registration_Overlay(img_base, img_transformed, intensity, path, invert_transformed=False):
    """
    Generate a preview of registered images by displaying the base (refernce image) in the R channel and the transformed
    image in the BG channel
    :param img_base: reference image
    :param img_transformed: target image
    :param intensity: intensity of target image (0->1)
    :param path: save path of preview image [*.png]
    :param invert_transformed: 'True' if the target image is to be inverted (255-image)
    :return: (none)
    """
    if invert_transformed:
        img_transformed = 255.0 - img_transformed

    img_reg_preview = np.zeros((img_base.shape[0], img_base.shape[1], 3))
    if len(img_base.shape) == 3:
        img_base = np.mean(np.array(img_base), 2)
    if len(img_transformed.shape) == 3:
        img_transformed = np.mean(np.array(img_transformed), 2)
    img_reg_preview[:, :, 0] = img_base / np.max(img_base)
    img_reg_preview[:, :, 1] = intensity * (img_transformed / np.max(img_transformed))
    img_reg_preview[:, :, 2] = intensity * (img_transformed / np.max(img_transformed))
    img_reg_preview = (img_reg_preview * 255.0).astype('uint8')
    if path is None:
        plt.imshow(img_reg_preview)
        plt.show()
    else:
        cv2.imwrite(path, cv2.cvtColor(img_reg_preview, cv2.COLOR_RGB2BGR))
    return img_reg_preview


def Preview_Keypoints(img_source, kp, path):
    """
    Save or display a preview of an image along with the detected keypoints
    :param img_source: image
    :param kp: keypoints
    :param path: save path of image. If 'None', will open a display image
    :return: (none)
    """
    img_prev_kp = np.copy(img_source)
    if len(img_prev_kp.shape) == 3:
        img_prev_kp = ExtendToRGBchannels(img_prev_kp)

    img_prev_kp = cv2.drawKeypoints(img_source, kp, img_prev_kp, color=(0, 255, 0), flags=0)
    if path is None:
        plt.imshow(img_prev_kp)
        plt.show()
    else:
        cv2.imwrite(path, img_prev_kp)


def Preview_Matches(img1, img2, kp1, kp2, matches, path=None):
    """
    Display and save a preview of matches between reference and target image
    :param img1: reference image
    :param img2: target image
    :param kp1: keypoints for reference image
    :param kp2: keypoints for target image
    :param matches: matches between keypoints
    :param path: save path of image. If 'None', will open a display image
    :return: (none)
    """
    img_matches = np.empty((max(img2.shape[0], img1.shape[0]), img2.shape[1] + img1.shape[1], 3),
                           dtype=np.uint8)

    cv2.drawMatches(img1.astype('uint8'), kp1, img2.astype('uint8'), kp2, matches, img_matches,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if path is None:
        plt.imshow(img_matches)
        plt.show()
    else:
        cv2.imwrite(path, img_matches)


def Crop_Image_Window(img, window_size, window_center):
    """
    Crop an image to a specified window size and center
    :param img: image array
    :type img: numpy array [W,H]
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
    return img[img_start_idx[1]:img_end_idx[1], img_start_idx[0]:img_end_idx[0]]


def Reframe_Image(img, new_scale, new_offset, frame_size, img_scaled=None):
    """
    Reframe image by rescaling and translating across a larger canvas
    :param img: image to be reframes
    :type img: numpy array [W,H,3]
    :param new_scale: Fractional X, Y scale (0->1) to be applied to img
    :type new_scale: numpy array [2]
    :param new_offset: X, Y translation in pixels for positioning the img in the canvas
    :type new_offset: numpy array [2]
    :param frame_size: canvas dimensions (pixels) for reframining
    :type frame_size: numpy array [2]
    :return: img_framed: reframed image with scale and translation applied to img
        img_scaled: img after rescaling only
    :rtype:
    """

    if len(np.shape(img))>2:
        img = np.mean(np.copy(img), 2)

    if img_scaled is None:
        img_scaled = cv2.resize(img, (int(img.shape[1] * new_scale[1]), int(img.shape[0] * new_scale[0])))
    img_framed = np.zeros(frame_size)
    img_start_idx = [int(new_offset[0] - img_scaled.shape[1] / 2), int(new_offset[1] - img_scaled.shape[0] / 2)]
    img_end_idx = [int(new_offset[0] + img_scaled.shape[1] / 2), int(new_offset[1] + img_scaled.shape[0] / 2)]
    if img_end_idx[1] >= frame_size[0]:
        img_end_idx[1] = frame_size[0] - 1
    if img_end_idx[0] >= frame_size[1]:
        img_end_idx[0] = frame_size[1]
    if img_start_idx[1] < 0 or img_start_idx[1] > frame_size[0] or img_start_idx[0] < 0 or img_start_idx[0] > frame_size[1]:
        print("Registration_src error: start/end indexes outside of image frame")
        print("Start index: {}, end index: {}, frame size: {}".format(img_start_idx, img_end_idx, frame_size))
        return img_framed
    else:
        img_crop = img_scaled.copy()
        display_crop_size = img_scaled.shape
        frame_crop_size = [(img_end_idx[1] - img_start_idx[1]), (img_end_idx[0] - img_start_idx[0])]
        if (img_end_idx[1] - img_start_idx[1]) < display_crop_size[0]:
            img_crop = img_crop[0:(img_end_idx[1] - img_start_idx[1]), :]
        if (img_end_idx[0] - img_start_idx[0]) < display_crop_size[1]:
            img_crop = img_crop[:, 0:(img_end_idx[0] - img_start_idx[0])]

        # compose rescaled image in frame:
        img_framed[img_start_idx[1]:img_end_idx[1], img_start_idx[0]:img_end_idx[0]] = img_crop

    return img_framed, img_scaled


def Preview_Initial_Alignment(img_FA, img_SLO, align_scale, align_pos, save_path=None):
    """
    Show a channel composite preview of intial alignment images
    :param img_FA: FA image
    :type img_FA: numpy array [W1,H1]
    :param img_SLO: SLO image
    :type img_SLO: numpy array [W2,H2]
    :param align_scale: alignment scale (X,Y)
    :type align_scale: numpy array [2]
    :param align_pos: alignment offset (X,Y) in pixels
    :type align_pos: numpy array [2]
    :param save_path: save image path. If 'None', show pyplot image plot
    :type save_path: string
    :return: none
    :rtype:
    """
    img_b_framed, img_b_scaled = Reframe_Image(img=img_SLO, new_scale=align_scale, new_offset=align_pos,
                                               frame_size=img_FA.shape)
    comp = CompositeChannelImages(img_FA, img_b_framed, b_intensity=1)
    if save_path is None:
        plt.figure(figsize=(10, 10))
        plt.imshow(comp)
        plt.show()
    else:
        comp = cv2.cvtColor((comp/np.max(comp)*255.0).astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, comp)
    return comp


def save_comparison_fig(ref_image, target_image, width, path):
    '''
    Save side-by-side comparison image
    :param ref_image: left image
    :param target_image: right image
    :param width: image size (for text display)
    :param path: output path
    :return: (none)
    '''
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(ref_image, cmap='gray')
    ax[1].imshow(target_image, cmap='gray')
    ax[0].text(10, width - 30, "FA", fontsize=15, color='white', bbox=dict(facecolor='Blue', alpha=0.5), )
    ax[1].text(width, width - 30, "SLO", fontsize=15, color='white', bbox=dict(facecolor='Blue', alpha=0.5), )
    plt.savefig(path)
    plt.close()

