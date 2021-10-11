import numpy as np
import cv2

fname = "close_calibration_cam1.npz"
with np.load(fname) as data:
    mtx = data['mtx']
    dist = data['dist']
    newcameramtx = data['newcameramtx']
    print(mtx)
    print(dist)

    # Just a test
    img = cv2.imread("./calibration_images/close_cam1/image0000.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    # all the docs say to use alpha=1. This is a terrible idea, as you will
    # see black borders. Use alpha=0 to crop to *just* the calibrated view
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst = cv2.resize(dst, (w//3,h//3))
    cv2.imshow("undistort", dst)
    img = cv2.resize(img, (w//3, h//3))
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
