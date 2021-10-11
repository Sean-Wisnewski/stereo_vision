"""
Runs one or two cameras using calibration matrices for each camera
"""
import numpy as np
import cv2
import argparse, sys

# TODO explore using libargus + cython to capture the input frames in cpp and do the rest of the processing in python
class CameraCapture:
    def __init__(self, sensor_id, calibration_fname):
        GSTREAMER_PIPELINE = f'nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=30/1 ' \
                             '! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! ' \
                             'videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        self.data = np.load(calibration_fname)
        self.mtx = self.data['mtx']
        self.dist = self.data['dist']
        self.newcameramtx = self.data['newcameramtx']

def capture_frames(sensor_id, calibration_fname):
    print(f"Starting camera {sensor_id}")
    camera = CameraCapture(sensor_id, calibration_fname)
    while True:
        key, frame = camera.cap.read()
        if not key:
            break
        cv2.imshow(f"Cam {sensor_id}", frame)
        undist = cv2.undistort(frame, camera.mtx, camera.dist, None, camera.newcameramtx)
        cv2.imshow(f"Cam {sensor_id}(Undistorted)", undist)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cam0', action="store_true", help="Run the first camera")
    parser.add_argument('-cam0_calibration', type=str, help="Path to file containing the calibration parameters for cam0")
    parser.add_argument('-cam1', action="store_true", help="Run the second camera")
    parser.add_argument('-cam1_calibration', type=str, help="Path to file containing the calibration parameters for cam1")

    return parser

def handle_cam_starting(args, cam_num, calibration_fname):
    if calibration_fname is not None:
        print("cam")
    else:
        sys.stderr.write(f"Failed to start cam{cam_num}: no calibration file provided\n")

def main():
    parser = make_argparser()
    args = parser.parse_args()
    if args.cam0:
        handle_cam_starting(args, 0, args.cam0_calibration)
    if args.cam1:
        handle_cam_starting(args, 1, args.cam1_calibration)

    test = 5
if __name__=="__main__":
    main()