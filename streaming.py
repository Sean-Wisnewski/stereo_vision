"""
Code pertinent to streaming from two cameras simultaneously
"""

import cv2


def capture_frames():


    # god this is ugly
    gst_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw ! jpegenc ! image/jpeg ! appsink"
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

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