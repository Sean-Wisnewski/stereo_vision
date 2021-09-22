"""
Code pertinent to streaming from two cameras simultaneously
"""

import cv2
import threading

class CameraCapture:
    def __init__(self, sensor_id):
        GSTREAMER_PIPELINE = f'nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=30/1 ' \
                             '! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! ' \
                             'videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)


def capture_frames(sensor_id=0):
    # The magic pipeline to get things to work, I have literally no idea how this works but it captures video from the csi cameras so woo
    #GSTREAMER_PIPELINE = 'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'

    camera = CameraCapture(0)
    while True:
        return_key, frame = camera.cap.read()
        if not return_key:
            break
        cv2.imshow(f'Cam {sensor_id}', frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def main():
    cam_0 = threading.Thread(target=capture_frames, args=(0,))
    cam_1 = threading.Thread(target=capture_frames, args=(1,))
    cams = []
    for idx in range(2):
        cam = threading.Thread(target=capture_frames, args=(idx,))
        cams.append(cam)
        cam.start()
    for idx, cam in enumerate(cams):
        cam.join()

if __name__=="__main__":
    main()