import cv2
import numpy as np
import argparse

class NanoCalibratedCameraCapture:
    def __init__(self, sensor_id, calibration_fname):
        GSTREAMER_PIPELINE = f'nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=21/1 ' \
                             '! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! ' \
                             'videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        self.data = np.load(calibration_fname)
        self.mtx = self.data['mtx']
        self.dist = self.data['dist']
        self.newcameramtx = self.data['newcameramtx']
        # magic numbers bad but here we are
        scale_x = 960/3264
        scale_y = 616/2464
        # scale fx, cx, fy, cy to match input resolution, not calibration resolution
        self.mtx[0,0]*=scale_x
        self.mtx[0,2]*=scale_x
        self.mtx[1,1]*=scale_y
        self.mtx[1,2]*=scale_y

    def read_frame(self, make_gray=False):
        ret, frame = self.cap.read()
        if make_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return ret, frame

def make_stereo_matcher():
    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = -128
    max_disp = 128
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

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('cam0_calibration', type=str, help="Path to file containing the calibration parameters for cam0")
    parser.add_argument('cam1_calibration', type=str, help="Path to file containing the calibration parameters for cam1")

    return parser

def main():
    # TODO maybe make these not hardcoded later?
    parser = make_argparser()
    args = parser.parse_args()
    cam0 = NanoCalibratedCameraCapture(0, args.cam0_calibration)
    cam1 = NanoCalibratedCameraCapture(1, args.cam1_calibration)
    while True:
        key0, gray0 = cam0.read_frame(make_gray=True)
        key1, gray1 = cam1.read_frame(make_gray=True)
        cv2.imshow("Cam 0", gray0)
        cv2.imshow("Cam 1", gray1)
        undist0 = cv2.undistort(gray0, cam0.mtx, cam0.dist, None)
        undist1 = cv2.undistort(gray1, cam1.mtx, cam1.dist, None)
        cv2.imshow("Cam 0(undist)", undist0)
        cv2.imshow("Cam 1(undist)", undist1)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

main()