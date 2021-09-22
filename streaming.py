"""
Code pertinent to streaming from two cameras simultaneously
"""

import cv2

class CameraCapture:
    def __init__(self, sensor_id):
        GSTREAMER_PIPELINE = f'nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=30/1 ' \
                             '! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! ' \
                             'videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)


def capture_frames(sensor_id=0):


    # The magic pipeline to get things to work, I have literally no idea how this works but it captures video from the csi cameras so woo
    #GSTREAMER_PIPELINE = 'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'

    camera0 = CameraCapture(sensor_id=0)
    camera1 = CameraCapture(sensor_id=1)
    while True:
        return_key0, frame0 = camera0.cap.read()
        return_key1, frame1 = camera1.cap.read()
        if not return_key0:
            break
        cv2.imshow('Cam 0', frame0)
        cv2.imshow('Cam 1', frame1)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def main():
    capture_frames()

if __name__=="__main__":
    main()