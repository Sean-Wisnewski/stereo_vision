import cv2
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# Test file to try and make a depth map for two input images
# will get converted to video code later
# code adapted from this tutorial: https://www.andreasjakl.com/easily-create-depth-maps-with-smartphone-ar-part-1/
fname1 = "imgs_for_code/sample_rect_imgs/image0000.jpg"
fname2 = "imgs_for_code/sample_rect_imgs/image0001.jpg"
img1 = cv2.imread(fname1)
img2 = cv2.imread(fname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#cv2.imshow("one", gray1)
#cv2.imshow("two", gray2)

# Show the input images
#fig, axes = plt.subplots(1, 2, figsize=(15,10))
#axes[0].imshow(gray1, cmap="gray")
#axes[1].imshow(gray2, cmap="gray")
#axes[0].axhline(250)
#axes[1].axhline(250)
#axes[0].axhline(450)
#axes[1].axhline(450)
#plt.suptitle("Original Images")
#plt.show()

# Find keypoints in both images
sift = cv2.SIFT_create()
fast = cv2.FastFeatureDetector_create()
kp1 = fast.detect(gray1, None)
kp2 = fast.detect(gray2, None)

freak = cv2.xfeatures2d.FREAK_create()
kp1, des1 = freak.compute(gray1, kp1)
kp2, des2 = freak.compute(gray2, kp2)
des1 = des1.astype(np.float32)
des2 = des2.astype(np.float32)

imgSift = cv2.drawKeypoints(gray1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("SIFT Keypoints", imgSift)
#cv2.waitKey(0)


# Use FLANN to match keypoints in both images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# keep only the good matches
matches_mask = [[0,0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matches_mask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Visualize some points
# TODO this is fucked, so fix it - kps are not matching up correctly at all basically
# best solution might be to make the background not crap
draw_params = dict(matchColor=(0,255,0), singlePointColor=(255,0,0),
                   matchesMask=matches_mask[0:500], flags=cv2.DrawMatchesFlags_DEFAULT)
keypoint_matches = cv2.drawMatchesKnn(gray1, kp1, gray2, kp2, matches[0:500], None, **draw_params)
#cv2.imshow("Keypoint Matches", keypoint_matches)
#cv2.waitKey(0)

# Rectification

# Find fundamental matrix for cameras
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fnd_mtx, inliners = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

pts1 = pts1[inliners.ravel() == 1]
pts2 = pts2[inliners.ravel() == 1]



# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, fnd_mtx)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(gray1, gray2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, fnd_mtx)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(gray2, gray1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
#plt.show()

# Stereo Rectification (Uncalibrated)
h1, w1 = gray1.shape
h2, w2 = gray2.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fnd_mtx, imgSize=(w1, h1))

# undistort images
gray1_rectified = cv2.warpPerspective(gray1, H1, (w1,h1))
gray2_rectified = cv2.warpPerspective(gray2, H2, (w2,h2))
# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(gray1_rectified, cmap="gray")
axes[1].imshow(gray2_rectified, cmap="gray")
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Rectified images")
#plt.savefig("rectified_images.png")
#plt.show()


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

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
plt.imshow(disparity_SGBM)
plt.show()
#cv2.imshow("Disparity", disparity_SGBM)
#cv2.imshow("gray1", gray1)
#cv2.imshow("gray2", gray2)
#cv2.waitKey(0)
#cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM)