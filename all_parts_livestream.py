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

def rectify_imgs(img1, img2):
    gray1 = change_to_gray(img1)
    gray2 = change_to_gray(img2)
    kp1, des1, _ = find_keypoints(gray1)
    kp2, des2, _ = find_keypoints(gray2)
    matches = match_keypoints(des1, des2)
    matches_mask, pts1, pts2 = filter_matches(matches, kp1, kp2)
    fnd_mtx, pts1, pts2 = find_fund_mtx(pts1, pts2)
    H1, H2 = stereo_rectification(gray1, gray2, pts1, pts2, fnd_mtx)
    gray1_rect, gray2_rect = undistort_imgs(gray1, gray2, H1, H2)
    return gray1_rect, gray2_rect

####################
# Control Flow
####################

def start_run(runtype, idx0=None, idx1=None, cal0=None, cal1=None):
    if runtype == "Uncalibrated":
        print("Uncal")
    elif runtype == "Calibrated":
        print("Cal")
    elif runtype == "Rectified":
        print("Rect")
    elif runtype == "DFD":
        print("DFD")
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
        gray_rect1, gray_rect2 = rectify_imgs(frame1, frame2)
        recolored_1 = recolor_img(gray_rect1)
        recolored_2 = recolor_img(gray_rect2)
        cv2.imshow(f"Camera {cam0.idx}(Rectified)", recolored_1)
        cv2.imshow(f"Camera {cam1.idx}(Rectified)", recolored_2)
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
        gray_rect1, gray_rect2 = rectify_imgs(frame1, frame2)
        dfd_img = depth_from_disparity(gray_rect1, gray_rect2)
        cv2.imshow("DFD Image", dfd_img)
        as_tensor = preprocess_image(dfd_img)
        output_dict = inference_for_single_image(model, as_tensor)
        output_dict = filter_unconfident_predictions(output_dict, 0.4)
        with_boxes = draw_bounding_boxes(dfd_img, output_dict, give_annotated=True, colors=colors)
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
    start_run(args.runtype)


if __name__=="__main__":
    main()
