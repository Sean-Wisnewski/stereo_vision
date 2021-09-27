import numpy as np
import cv2

fname = "cam1.npz"
with np.load(fname) as data:
    mtx = data['mtx']
    dist = data['dist']
    print(mtx)
    print(dist)

    # Just a test
    img = cv2.imread("./calibration_images/cam1/image0020.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    dst = cv2.resize(dst, (w//3,h//3))
    cv2.imshow("undistort", dst)
    img = cv2.resize(img, (w//3, h//3))
    cv2.imshow("original", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()