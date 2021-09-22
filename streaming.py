"""
Code pertinent to streaming from two cameras simultaneously
"""

import cv2


def capture_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    while True:
        return_key, frame = cap.read()
        if not return_key:
            break
        cv2.imshow('Cam 0', frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def main():
    capture_frames()

if __name__=="__main__":
    main()