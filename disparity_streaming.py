import cv2
import numpy as np
from mobilenet_livestream import NanoCameraCapture


"""
class NanoCameraCapture:
    def __init__(self, sensor_id, calibration_fname=None):
        GSTREAMER_PIPELINE = f'nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=30/1 ' \
                             '! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! ' \
                             'videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        self.idx = sensor_id
        if calibration_fname is not None:
            self.data = np.load(calibration_fname)
            self.mtx = self.data['mtx']
            self.dist = self.data['dist']
            self.newcameramtx = self.data['newcameramtx']

    def capture_frame_cb(self):
        ret, frame = self.cap.read()
        return ret, frame
"""

def change_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def find_keypoints(gray):
    # this is not included with cv2 directly b/c of patents and I don't currently feel like fighting dependency hell
    # well I ended up needed opencv-contrib-python for both, so recompiling from source *should* let this work
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    # need to install `pip install opencv-contrib-python to get this
    # so, replace it with something else
    #freak = cv2.xfeatures2d.FREAK_create()
    # need to do keypoint detection with something else b/c detect and compute is unimplemented for FREAK
    #fast = cv2.FastFeatureDetector_create()
    #kp = fast.detect(gray, None)
    #kp, des = freak.detectAndCompute(gray, None)
    #kp, des = freak.compute(gray, kp)
    # convert to a dtype accepted by flann later
    #des = des.astype(np.float32)
    img_sift = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img_sift

def match_keypoints(des1, des2):
    flann_index_kdtree = 1
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def filter_matches(matches, kp1, kp2):
    matches_mask = [[0,0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matches_mask[i] = [1,0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    return matches_mask, pts1, pts2

def draw_kp_matches(matches, matches_mask, gray1, gray2, kp1, kp2):
    draw_params = dict(matchColor=(0,255,0), singlePointColor=(255,0,0),
                       matchesMask=matches_mask[0:500], flags=cv2.DrawMatchesFlags_DEFAULT)
    keypoint_matches = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches[0:500], None, **draw_params)
    cv2.imshow("Keypoint Matches", keypoint_matches)
    cv2.waitKey(0)
    
def find_fund_mtx(pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fnd_mtx, inliners = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    pts1 = pts1[inliners.ravel() == 1]
    pts2 = pts2[inliners.ravel() == 1]
    return fnd_mtx, pts1, pts2

def stereo_rectification(gray1, gray2, pts1, pts2, fnd_mtx):
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fnd_mtx, imgSize=(w1, h1))
    return H1, H2

def undistort_imgs(gray1, gray2, H1, H2):
    h1, w1 = gray1.shape
    h2, w2 = gray2.shape
    gray1_rectified = cv2.warpPerspective(gray1, H1, (w1, h1))
    gray2_rectified = cv2.warpPerspective(gray2, H2, (w2, h2))
    return gray1_rectified, gray2_rectified

def depth_from_disparity(gray1_rectified, gray2_rectified):
    # ------------------------------------------------------------
    # CALCULATE DISPARITY (DEPTH MAP)
    # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
    # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

    # StereoSGBM Parameter explanations:
    # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

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
    disparity_SGBM = stereo.compute(gray1_rectified, gray2_rectified)
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
    return np.uint8(disparity_SGBM)

def main():
    cam0 = NanoCameraCapture(0)
    cam1 = NanoCameraCapture(1)
    # TODO clean all this up to be cleaner
    while True:
        ret1, frame1 = cam0.capture_frame_cb()
        ret2, frame2 = cam1.capture_frame_cb()
        gray1 = change_to_gray(frame1)
        gray2 = change_to_gray(frame2)
        kp1, des1, _ = find_keypoints(gray1)
        kp2, des2, _ = find_keypoints(gray2)
        matches = match_keypoints(des1, des2)
        matches_mask, pts1, pts2 = filter_matches(matches, kp1, kp2)
        # skipping showing matches for now...
        fnd_mtx, pts1, pts2 = find_fund_mtx(pts1, pts2)
        H1, H2 = stereo_rectification(gray1, gray2, pts1, pts2, fnd_mtx)
        gray1_rect, gray2_rect = undistort_imgs(gray1, gray2, H1, H2)
        dfd_img = depth_from_disparity(gray1_rect, gray2_rect)
        cv2.imshow("DFD map", dfd_img)
        if cv2.waitKey(1) == 27:
            break
    cam0.cap.release()
    cam1.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



