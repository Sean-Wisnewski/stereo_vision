"""
Code pertinent to streaming from two cameras simultaneously
"""

import cv2


def capture_frames():


    # god this is ugly
    gst_pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw ! jpegenc ! image/jpeg ! appsink"
    GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

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