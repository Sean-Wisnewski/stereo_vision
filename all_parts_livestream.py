import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import argparse, sys

from mobilenet_livestream import *
from disparity_streaming import *

####################
# Helper functions
####################

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('runtype', type=str, choices=["Uncalibrated", "Calibrated", "Rectified", "DFD"],
                        help="Type of run to use")
    parser.add_argument('labels_fname', type=str, help="File containing the class labels for Mobilenet")
    parser.add_argument('colors_fname', type=str, help="File containing the array of colors to use for each class in mobilenet")
    parser.add_argument('--idx0', type=int, help="Index of the first camera")
    parser.add_argument('--idx1', type=int, help="Index of the second camera")
    parser.add_argument('--cal_file0', type=str, help="Filename containing the calibration matrix for the first camera")
    parser.add_argument('--cal_file1', type=str, help="Filename containing the calibration matrix for the second camera")
    parser.add_argument('--model', type=str, help="path to directory containing the model OR model name on tf hub",
                        default="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
    return parser

def check_args(args):
    if args.runtype == "Uncalibrated":
        if args.idx0 is None:
            sys.stderr.write(f"Error: for Uncalibrated, idx0 must be specified\n")
            exit(-1)
    if args.runtype == "Calibrated":
        if args.idx0 is None:
            sys.stderr.write(f"Error: for Calibrated, idx0 must be specified\n")
            exit(-1)
        if args.cal_file0 is None:
            sys.stderr.write(f"Error: for Calibrated, a calibration file (cal_file0) must be specified\n")
            exit(-1)
    if args.runtype == "Rectified" or args.runtype=="DFD":
        if args.idx0 is None:
            sys.stderr.write(f"Error: for Rectified/DFD, idx0 must be specified\n")
            exit(-1)
        if args.cal_file0 is None:
            sys.stderr.write(f"Error: for Rectified/DFD, a calibration file (cal_file0) must be specified\n")
            exit(-1)
        if args.idx1 is None:
            sys.stderr.write(f"Error: for Rectified/DFD, idx1 must be specified\n")
            exit(-1)
        if args.cal_file1 is None:
            sys.stderr.write(f"Error: for Rectified/DFD, a calibration file (cal_file1) must be specified\n")
            exit(-1)

def recolor_img(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def make_stereo_matcher():
    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = 0
    max_disp = 256
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 0
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    return stereo

def compute_dfd_map(stereo, imgL, imgR, normalize=False):
    disparty_SGBM = stereo.compute(imgL, imgR)
    if normalize:
        disparty_SGBM = cv2.normalize(disparty_SGBM, disparty_SGBM, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    # since we normalize to 0..255, return a uint8 matrix
    return np.uint8(disparty_SGBM)

def rectify_imgs(img0, img1, cam0, cam1, recolor=False):
    """
    TODO figure out wth is wrong with stereoRectify b/c it just don't wanna work
    :param img0:
    :param img1:
    :param cam0:
    :param cam1:
    :param recolor:
    :return:
    """
    gray1 = change_to_gray(img0)
    gray2 = change_to_gray(img1)
    undist0 = cv2.undistort(img0, cam0.mtx, cam0.dist, None)
    undist1 = cv2.undistort(img1, cam1.mtx, cam1.dist, None)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam0.mtx, cam0.dist, cam1.mtx, cam1.dist, (616, 960, 3), None, None)
    undist_map0, rect_tranform_map0 = cv2.initUndistortRectifyMap(cam0.mtx, cam0.dist, R1, None, (616, 960), None)
    undist_map1, rect_tranform_map1 = cv2.initUndistortRectifyMap(cam1.mtx, cam1.dist, R2, None, (616, 960), None)
    rect0 = cv2.remap(undist0, undist_map0, rect_tranform_map0, cv2.INTER_LINEAR)
    rect1 = cv2.remap(undist1, undist_map1, rect_tranform_map1, cv2.INTER_LINEAR)
    return rect0, rect1

def make_dfd_map(img0, img1, cam0, cam1):
    gray1 = change_to_gray(img0)
    gray2 = change_to_gray(img1)
    undist0 = cv2.undistort(gray1, cam0.mtx, cam0.dist, None)
    undist1 = cv2.undistort(gray2, cam1.mtx, cam1.dist, None)
    stereo = make_stereo_matcher()
    dfd = compute_dfd_map(stereo, undist0, undist1, normalize=True)
    return dfd


####################
# Control Flow
####################

def start_run(runtype, model=None, idx0=None, idx1=None, cal0=None, cal1=None, class_dict=None, colors=None):
    if runtype == "Uncalibrated":
        print("Uncal")
        # TODO change to use whichever of idx0 or idx1 is not none
        uncalibrated_run(model, idx0, class_dict, colors)
    elif runtype == "Calibrated":
        print("Cal")
        # TODO change to use whichever of idx0 or idx1 is not none
        calibrated_run(model, idx0, cal0, class_dict, colors)
    elif runtype == "Rectified":
        print("Rect")
        rectified_run(model, idx0, idx1, cal0, cal1, class_dict, colors)
    elif runtype == "DFD":
        print("DFD")
        dfd_run(model, idx0, idx1, cal0, cal1, class_dict, colors)
    else:
        print("Literally how did you get here")


def uncalibrated_run(model, idx0, class_dict, colors):
    """
    Controls the running and display of frames + annotations of a single uncalibrated camera
    :param idx0:
    :param class_dict:
    :param colors:
    :return:
    """
    cam = NanoCameraCapture(idx0)
    while True:
        ret, frame = cam.capture_frame_cb()
        cv2.imshow(f"Camera {cam.idx}", frame)
        as_tensor = preprocess_image(frame)
        output_dict = inference_for_single_image(model, as_tensor)
        output_dict = filter_unconfident_predictions(output_dict, 0.4)
        with_boxes = draw_bounding_boxes(frame, output_dict, give_annotated=True, colors=colors)
        draw_bounding_boxes_with_labels_confidence(with_boxes, output_dict, class_dict, colors=colors)
        if cv2.waitKey(1) == 27:
            break
    cam.cap.release()
    cv2.destroyAllWindows()


def calibrated_run(model, idx0, cal_fname, class_dict, colors):
    """
    :param idx0:
    :param cal_mtx:
    :param class_dict:
    :param colors:
    :return:
    """
    cam = NanoCameraCapture(idx0, cal_fname)
    while True:
        ret, frame = cam.capture_frame_cb()
        cv2.imshow(f"Camera {cam.idx}(Calibrated)", frame)
        undist = cv2.undistort(frame, cam.mtx, cam.dist, None)
        cv2.imshow(f"Camera {cam.idx}(Calibrated)", undist)
        as_tensor = preprocess_image(frame)
        output_dict = inference_for_single_image(model, as_tensor)
        output_dict = filter_unconfident_predictions(output_dict, 0.4)
        with_boxes = draw_bounding_boxes(frame, output_dict, give_annotated=True, colors=colors)
        draw_bounding_boxes_with_labels_confidence(with_boxes, output_dict, class_dict, colors=colors)
        if cv2.waitKey(1) == 27:
            break
    cam.cap.release()
    cv2.destroyAllWindows()

def rectified_run(model, idx0, idx1, cal_fname0, cal_fname1, class_dict, colors):
    """
    :param idx0:
    :param idx1:
    :param cal_mtx0:
    :param cal_mtx1:
    :param class_dict:
    :param colors:
    :return:
    """
    cam0 = NanoCameraCapture(idx0, cal_fname0)
    cam1 = NanoCameraCapture(idx1, cal_fname1)
    while True:
        ret1, frame1 = cam0.capture_frame_cb()
        ret2, frame2 = cam1.capture_frame_cb()
        gray_rect1, gray_rect2 = rectify_imgs(frame1, frame2, cam0, cam1)
        recolored_1 = recolor_img(gray_rect1)
        recolored_2 = recolor_img(gray_rect2)
        cv2.imshow(f"Camera {cam0.idx}(Rectified)", recolored_1)
        cv2.imshow(f"Camera {cam1.idx}(Rectified)", recolored_2)
        """
        as_tensor1 = preprocess_image(recolored_1)
        as_tensor2 = preprocess_image(recolored_2)
        output_dict1 = inference_for_single_image(model, as_tensor1)
        output_dict1 = filter_unconfident_predictions(output_dict1, 0.4)
        output_dict2 = inference_for_single_image(model, as_tensor2)
        output_dict2 = filter_unconfident_predictions(output_dict2, 0.4)
        with_boxes1 = draw_bounding_boxes(recolored_1, output_dict1, give_annotated=True, colors=colors)
        with_boxes2 = draw_bounding_boxes(recolored_2, output_dict2, give_annotated=True, colors=colors)
        draw_bounding_boxes_with_labels_confidence(with_boxes1, output_dict1, class_dict, colors=colors)
        draw_bounding_boxes_with_labels_confidence(with_boxes2, output_dict2, class_dict, colors=colors)
        """
        if cv2.waitKey(1) == 27:
            break
    cam0.cap.release()
    cam1.cap.release()
    cv2.destroyAllWindows()

def dfd_run(model, idx0, idx1, cal_fname0, cal_fname1, class_dict, colors):
    """
    Note: this is quite literally rectified, but with one more step to create the DFD map, then run inference on that
    :param idx0:
    :param idx1:
    :param cal_mtx0:
    :param cal_mtx1:
    :param class_dict:
    :param colors:
    :return:
    """
    cam0 = NanoCameraCapture(idx0, cal_fname0)
    cam1 = NanoCameraCapture(idx1, cal_fname1)
    while True:
        ret1, frame1 = cam0.capture_frame_cb()
        ret2, frame2 = cam1.capture_frame_cb()
        dfd = make_dfd_map(frame1, frame2, cam0, cam1)
        cv2.imshow(f"DFD Map", dfd)
        # add a depth dimension since the dfd map will only have 2 dimensions, and we need at least 3 to resize it
        as_tensor = preprocess_image(dfd)
        output_dict = inference_for_single_image(model, as_tensor)
        output_dict = filter_unconfident_predictions(output_dict, 0.4)
        with_boxes = draw_bounding_boxes(dfd, output_dict, give_annotated=True, colors=colors)
        draw_bounding_boxes_with_labels_confidence(with_boxes, output_dict, class_dict, colors=colors)
        if cv2.waitKey(1) == 27:
            break
    cam0.cap.release()
    cam1.cap.release()
    cv2.destroyAllWindows()


def main():
    parser = make_argparser()
    args = parser.parse_args()
    check_args(args)
    class_labels_dict = load_pkl_file(args.labels_fname)
    colors = load_pkl_file(args.colors_fname)
    model = load_from_hub(args.model)

    start_run(args.runtype, model, args.idx0, args.idx1, args.cal_file0, args.cal_file1, class_labels_dict, colors)


if __name__=="__main__":
    main()
