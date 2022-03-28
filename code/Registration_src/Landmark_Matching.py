import numpy as np
import cv2
from Registration_src.registration_parameters import *
from Registration_src.Utils.Visualization import Preview_Matches
from matplotlib import pyplot as plt


def GPU_Orb():

    # Detect ORB features and compute descriptors.
    max_features = 500
    orb = cv2.ORB.create(max_features)
    image_keypoints, image_descriptors = orb.detectAndCompute(img_target, None)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(img_ref, None)


def FAST_FeatureDetection(img):
    """
    Detect feature coordinates using FAST algorithm (good features to detect)
    :param img:
    :return: list of detected feature coordinates
    """
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp


def AKAZE_HOG_Matching(img_ref, img_target, path=None):

    AKAZE = cv2.AKAZE_create(nOctaveLayers=7)
    kp1, d1 = AKAZE.detectAndCompute(img_ref, None)
    kp2, d2 = AKAZE.detectAndCompute(img_target, None)

    if len(kp1) < 3 or len(kp2) < 3:
        print("Warning: (AKAZE_HOG_Matching) Fewer than 3 landmarks for AKAZE. Skipping.")
        return kp1, kp2, []

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(d1, d2)
    # print("# of total matches: {}".format(len(matches)))
    list(matches).sort(key=lambda x: x.distance)
    matches = matches[:350]

    '''    f, ax = plt.subplots(1, 2)
            ax[0].imshow(img_ref, cmap='gray')
            ax[1].imshow(img_target, cmap='gray')
            ax[0].scatter(kp1[matches[0].queryIdx].pt[0], kp1[matches[0].queryIdx].pt[1], c='r')
            ax[1].scatter(kp2[matches[0].trainIdx].pt[0], kp2[matches[0].trainIdx].pt[1], c='r')
            plt.show()
    '''
    matches = Filter_Distant_Matches(kp1=kp1, kp2=kp2, matches=matches,
                                     distance_threshold=match_distance_filter_factor * img_ref.shape[0])

    #Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1, kp2=kp2, matches=matches,
    #                path=path + "matches_AKAZE.png")

    return kp1, kp2, matches


def ShiTomasi_HOG_Matching(img_ref, img_target, path=None):
    corners1 = cv2.goodFeaturesToTrack(image=img_ref, maxCorners=250, qualityLevel=0.01, minDistance=25, k=0.04)
    corners2 = cv2.goodFeaturesToTrack(image=img_target, maxCorners=250, qualityLevel=0.01, minDistance=25, k=0.04)
    corners1 = np.array(corners1[:,0,:])
    corners2 = np.array(corners2[:,0,:])
    # corners1 = np.int0(corners1)
    # corners2 = np.int0(corners2)

    '''    for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 3, 255, -1)
        plt.cla()
        plt.imshow(img)
        plt.scatter(corners[:, 0, 0], corners[:, 0, 1], c='r')
        plt.show()'''

    d_hog1 = HOG_Discription(img_ref, pt=list(corners1), window_size=(128, 128))
    d_hog2 = HOG_Discription(img_target, pt=list(corners2), window_size=(128, 128))

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(d_hog1, d_hog2)
    # print("# of total matches: {}".format(len(matches)))
    matches.sort(key=lambda x: x.distance)
    matches = matches[:250]

    '''    f, ax = plt.subplots(1,2)
        ax[0].imshow(img_ref, cmap='gray')
        ax[1].imshow(img_target, cmap='gray')
        ax[0].scatter(corners1[matches[0].queryIdx][0], corners1[matches[0].queryIdx][1], c='r')
        ax[1].scatter(corners2[matches[0].trainIdx][0], corners2[matches[0].trainIdx][1], c='r')
        plt.show()
    '''


    Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1_SIFT, kp2=kp2_SIFT, matches=matches_SIFT,
                    path=path + "matches_STHOG.png")

    matches = Filter_Distant_Matches(pt1=d_hog1, pt2=d_hog2, matches=matches,
                                     distance_threshold=match_distance_filter_factor * img_ref.shape[0])

    return d_hog1, d_hog2, matches


def HOG_Discription(img, kp=None, pt=None, window_size=HOG_window_size, block_size=HOG_block_size,
                    block_stride=HOG_block_stride, cell_size=HOG_cell_size):
    """
    Generate HOG (histogram) description of features
    :param img:
    :param pt:
    :param HOG_block_size:
    :param HOG_block_stride:
    :param HOG_cell_size:
    :return: list of feature matrix
    """
    if pt is None:
        pt = []
        for p in kp:
            pt.append((p.pt[0], p.pt[1]))

    hog = configure_HOG(winSize=window_size, blockSize=block_size, blockStride=block_stride,
                        cellSize=cell_size,
                        winSigma=2., nlevels=64)
    winStride = (8, 8)
    padding = (4, 4)
    hist_size = len(hog.compute(img, winStride, padding, np.array((pt[0],)).astype('int')))
    d_hog = hog.compute(img=img, winStride=winStride, padding=padding, locations=np.array(pt).astype('int')).reshape(len(pt),
                                                                                             hist_size)

    return d_hog


def FAST_HOG_Matching(img_ref, img_target):
    # obtain landmark coordinates
    kp1 = FAST_FeatureDetection(img_ref)
    kp2 = FAST_FeatureDetection(img_target)

    if len(kp1) < 3 or len(kp2) < 3:
        print("Warning: (FAST_HOG_Matching) Fewer than 3 landmarks for SIFT/HOG. Skipping.")
        return kp1, kp2, []

    # Compute HOG Descriptors
    d_hog1 = HOG_Discription(img_ref, kp1)
    d_hog2 = HOG_Discription(img_target, kp2)

    # Perform Matching
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = matcher.match(d_hog1, d_hog2)
    # print("# of total matches: {}".format(len(matches)))
    matches.sort(key=lambda x: x.distance)
    matches = matches[:350]

    matches = Filter_Distant_Matches(kp1=kp1, kp2=kp2, matches=matches,
                                     distance_threshold=match_distance_filter_factor * img_ref.shape[0])

    return kp1, kp2, matches


def SIFT_Matching(img_ref, img_target):
    # Get Sift Landmarks+Descriptors
    #sift = cv2.SIFT_create(contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma, edgeThreshold=SIFT_edgeThreshold)
    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(img_ref, None)
    kp2, d2 = sift.detectAndCompute(img_target, None)
    if d1 is None or d2 is None:
        print("Warning: (Registration_src) Fewer than 3 landmarks for SIFT/SIFT. Skipping.")
        return kp1, kp2, []
    else:
        d1 = (d1 * 255.0).astype('uint8')
        d2 = (d2 * 255.0).astype('uint8')
        # Perform Matching
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(d1, d2)
        # print("# of total matches: {}".format(len(matches)))
        list(matches).sort(key=lambda x: x.distance)
        matches = matches[:350]

        matches = Filter_Distant_Matches(kp1=kp1, kp2=kp2, matches=matches,
                                         distance_threshold=match_distance_filter_factor * img_ref.shape[0])
    return kp1, kp2, matches


def ORB_Matching(img_ref, img_target):
    # Get ORB Landmarks+Descriptors
    #orb_detector = cv2.ORB_create(ORB_count, patchSize=ORB_patchSize, nlevels=ORB_nlevels,
    #                              edgeThreshold=ORB_edgeThreshold, scoreType=cv2.ORB_FAST_SCORE)
    orb_detector = cv2.ORB_create()
    kp1, d1 = orb_detector.detectAndCompute(img_ref, None)
    kp2, d2 = orb_detector.detectAndCompute(img_target, None)
    # Perform Matching
    if d1 is not None and d2 is not None:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(d1, d2)
        list(matches).sort(key=lambda x: x.distance)
        matches = matches[:350]
    else:
        print("Warning: ORB landmark search failed. ref image landmarks is None: {},"
              " target image landmarks is None: {}".format(d1 is None, d2 is None))
        return kp1, kp2, []
    if len(matches) >= 10:
        matches = Filter_Distant_Matches(kp1=kp1, kp2=kp2, matches=matches,
                                         distance_threshold=match_distance_filter_factor * img_ref.shape[0])
    return kp1, kp2, matches


def configure_HOG(winSize=(64, 64), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8), nbins=9, derivAperture=1,
                  winSigma=4., histogramNormType=0, L2HysThreshold=2.e-01, gammaCorrection=0,
                  nlevels=64):
    """
    Initialize HOG description computer
    :param winSize: (see HOG documentation)
    :param blockSize: (see HOG documentation)
    :param blockStride: (see HOG documentation)
    :param cellSize: (see HOG documentation)
    :param nbins: (see HOG documentation)
    :param derivAperture: (see HOG documentation)
    :param winSigma: (see HOG documentation)
    :param histogramNormType: (see HOG documentation)
    :param L2HysThreshold: (see HOG documentation)
    :param gammaCorrection: (see HOG documentation)
    :param nlevels: (see HOG documentation)
    :return: hog object (cv2.HOGDescriptor)
    """
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    return hog


def Filter_Distant_Matches(matches, distance_threshold, kp1=None, kp2=None, pt1=None, pt2=None):
    """
    Steps through matches and remove matches between landmarks which have coordinate distances greater than
    'ditance_threshold'
    :param kp1: image 1 landmarks
    :param kp2: image 2 landmarks
    :param matches: list of matches
    :param distance_threshold: threshold (in pixels) for the maximum coordinate difference
    :return: filtered list of matches
    """
    matches_dst_filter = []
    for m in matches:
        if pt1 is None:
            coord1 = kp1[m.queryIdx].pt
            coord2 = kp2[m.trainIdx].pt
        else:
            coord1 = pt1[m.queryIdx]
            coord2 = pt2[m.trainIdx]
        dst = np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)
        if dst < distance_threshold:
            matches_dst_filter.append(m)
    return matches_dst_filter


def Landmark_Description_Set(img_ref, img_target, path=None):

    #ShiTomasi_HOG_Matching(img_ref, img_target)
    kp1_AKAZE, kp2_AKAZE, matches_AKAZE = AKAZE_HOG_Matching(img_ref, img_target)
    print("AKAZE: {} matches".format(len(matches_AKAZE)))
    #kp1_FAST, kp2_FAST, matches_FH = FAST_HOG_Matching(img_ref, img_target)
    matches_FH = None
    kp1_SIFT, kp2_SIFT, matches_SIFT = SIFT_Matching(img_ref, img_target)
    print("SIFT: {} matches".format(len(matches_SIFT)))
    kp1_ORB, kp2_ORB, matches_ORB = ORB_Matching(img_ref, img_target)
    print("ORB: {} matches".format(len(matches_ORB)))


    p1 = []
    p2 = []
    if matches_FH is not None:
        if path is not None:
            Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1_FAST, kp2=kp2_FAST, matches=matches_FH,
                            path=path + "matches_FAST-HOG.png")
        for i in range(len(matches_FH)):
            p1.append(kp1_FAST[matches_FH[i].queryIdx].pt)
            p2.append(kp2_FAST[matches_FH[i].trainIdx].pt)

    if matches_SIFT is not None:
        if path is not None:
            Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1_SIFT, kp2=kp2_SIFT, matches=matches_SIFT,
                            path=path + "matches_SIFT.png")
        for i in range(len(matches_SIFT)):
            p1.append(kp1_SIFT[matches_SIFT[i].queryIdx].pt)
            p2.append(kp2_SIFT[matches_SIFT[i].trainIdx].pt)
    if matches_ORB is not None:
        if path is not None:
            Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1_ORB, kp2=kp2_ORB, matches=matches_ORB,
                            path=path + "matches_ORB.png")
        for i in range(len(matches_ORB)):
            p1.append(kp1_ORB[matches_ORB[i].queryIdx].pt)
            p2.append(kp2_ORB[matches_ORB[i].trainIdx].pt)
    if matches_AKAZE is not None:
        if path is not None:
            Preview_Matches(img1=img_ref, img2=img_target, kp1=kp1_AKAZE, kp2=kp2_AKAZE, matches=matches_AKAZE,
                            path=path + "matches_AKAZE.png")
        for i in range(len(matches_AKAZE)):
            p1.append(kp1_AKAZE[matches_AKAZE[i].queryIdx].pt)
            p2.append(kp2_AKAZE[matches_AKAZE[i].trainIdx].pt)

    p1 = np.array(p1)
    p2 = np.array(p2)
    return p1, p2

