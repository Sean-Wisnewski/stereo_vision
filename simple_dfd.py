import cv2
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

fname1 = "imgs_for_code/sample_rect_imgs/image0000.jpg"
fname2 = "imgs_for_code/sample_rect_imgs/image0001.jpg"
cal1 = "final_matrices/close_calibration_cam0.npz"
cal2 = "final_matrices/close_calibration_cam1.npz"
img1 = cv2.imread(fname1)
img2 = cv2.imread(fname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

h1, w1 = gray1.shape
h2, w2 = gray2.shape
data1 = np.load(cal1)
mtx1 = data1['mtx']
dist1 = data1['dist']
newcammtx1 = data1['newcameramtx']
dst = cv2.undistort(gray1, mtx1, dist1, None, newcammtx1)
data2 = np.load(cal2)
mtx2 = data2['mtx']
dist2 = data2['dist']
newcammtx2 = data2['newcameramtx']
dst2 = cv2.undistort(gray2, mtx2, dist2, None, newcammtx2)


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

disparity_SGBM = stereo.compute(dst, dst2)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
plt.imshow(disparity_SGBM)
plt.show()