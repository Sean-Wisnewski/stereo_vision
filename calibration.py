"""
Used to calibrate a camera
Code adapted from https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html
"""
import numpy as np
import cv2
import glob, argparse
from tqdm import tqdm

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('calibration_dir', type=str, help="Name of directory containing calibration images")
    parser.add_argument('nrows', type=int, help="number of interior rows of chessboard pattern\ne.g. 10 rows = 9 interior")
    parser.add_argument('ncols', type=int, help="number of interior cols of chessboard pattern\ne.g. 8 rows = 7 interior")
    parser.add_argument('--output-file', type=str, help="Filename to save camera calibration matrices to")
    parser.add_argument('--img-ftype', type=str, help="Filetype of calibration images, default=jpg", default='.jpg')
    return parser

def make_arrays(input_shape):
    """
    Creates the objpt array and imgpt arrays to store calibration results for use later
    :param input_shape:
    :return:
    """
    objp = np.zeros((input_shape[0]*input_shape[1],3), np.float32)
    objp[:, :2] = np.mgrid[0:input_shape[0], 0:input_shape[1]].T.reshape(-1,2).astype(np.float32)

    # 3d points in real world space (Z will always be zero)
    objpts = []
    # 2d points in image plane
    imgpts = []
    return objp, objpts, imgpts

def read_all_images(dname):
    return glob.glob(f"{dname}/*.jpg")

def calibrate_camera(imgs, objp : np.ndarray, objpts : list, imgpts : list, input_shape):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for fname in tqdm(imgs):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, input_shape, None)

        # If found, add the object points and image points (after refining corners)
        if ret:
            objpts.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),criteria)
            imgpts.append(corners_refined)

            # Draw and optinally display the corners
            cv2.drawChessboardCorners(img, input_shape, corners_refined, ret)
            #cv2.imshow('Corners', img)
            #cv2.waitKey(1)
    # read the first image for shape
    img = cv2.imread(imgs[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
    return mtx, dist


def main():
    parser = make_argparser()
    args = parser.parse_args()

    input_shape = (args.nrows, args.ncols)
    objp, objpts, imgpts = make_arrays(input_shape)
    imgs = read_all_images(args.calibration_dir)
    mtx, dist = calibrate_camera(imgs, objp, objpts, imgpts, input_shape)
    if args.output_file is not None:
        np.savez(args.output_file, mtx=mtx, dist=dist)




if __name__ == "__main__":
    main()
