import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
from PIL import Image, ImageDraw, ImageFont
import pickle


class NanoCameraCapture:
    def __init__(self, sensor_id, calibration_fname=None):
        GSTREAMER_PIPELINE = f'nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=30/1 ' \
                             '! nvvidconv flip-method=0 ! video/x-raw, width=780, height=440, format=(string)BGRx ! ' \
                             'videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'
        self.cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
        self.idx = sensor_id
        if calibration_fname is not None:
            self.data = np.load(calibration_fname)
            self.mtx = self.data['mtx']
            self.dist = self.data['dist']
            self.newcameramtx = self.data['newcameramtx']
            # magic numbers bad but here we are
            #scale_x = 960/3264
            #scale_y = 616/2464
            scale_x = 780/3264
            scale_y = 440/2464
            # scale fx, cx, fy, cy to match input resolution, not calibration resolution
            self.mtx[0,0]*=scale_x
            self.mtx[0,2]*=scale_x
            self.mtx[1,1]*=scale_y
            self.mtx[1,2]*=scale_y

    def capture_frame_cb(self):
        ret, frame = self.cap.read()
        return ret, frame



class CameraCapture:
    def __init__(self, cam_idx):
        self.cap = cv2.VideoCapture(cam_idx)
        self.idx = cam_idx

    def capture_frames(self):
        """
        Captures and displays frames continuously
        :return:
        """
        while True:
            ret, frame = self.cap.read()
            cv2.imshow(f"Camera {self.idx}", frame)
            as_tensor = preprocess_image(frame)
            print(as_tensor.shape)
            if cv2.waitKey(1) == 27:
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def capture_frame_cb(self):
        """
        Captures and returns a single frame
        :return:
        """
        ret, frame = self.cap.read()
        return ret, frame

def load_from_hub(model_name):
    model = hub.load(model_name)
    return model

def load_pkl_file(fname):
    with open(fname, "rb") as handle:
        return pickle.load(handle)

def preprocess_image(image, img_size=tf.constant([224, 224])):
    """
    Converts an input image into the correct format for mobilenet
    :param image:
    :param img_size:
    :return:
    """
    if type(image) == np.array or type(image) == np.ndarray:
        input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    elif type(image) == tf.Tensor:
        input_tensor = tf.cast(image, tf.uint8)
    else:
        try:
            input_tensor = tf.cast(image, tf.uint8)
        except Exception:
            raise TypeError(f"Incorrect type for image provided: {type(image)}")
    if input_tensor.shape != (224, 224, 3):
        input_tensor = tf.image.resize(input_tensor, img_size)
    return input_tensor

def preprocess_input_map_fn(img, label):
    img = tf.cast(img, tf.uint8)
    """
    if len(img.shape) < 4:
        img = img[tf.newaxis, ...]
    elif len(img.shape) > 4:
        raise ValueError(f"Input image has too many dimensions: {img.shape}")
    """
    return img, label


def load_dataset(data_dir, bs=32, img_size=(224,224)):
    ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, shuffle=True, image_size=img_size,
                                                             batch_size=bs)
    ds = ds.map(preprocess_input_map_fn)
    return ds

def inference_for_single_image(model, input_tensor):
    # convert to uint8 if it's not already
    if input_tensor.dtype != tf.uint8:
        input_tensor = tf.cast(input_tensor, tf.uint8)
    # add the new axis if necessary
    if len(input_tensor.shape) < 4:
        input_tensor = input_tensor[tf.newaxis, ...]
    elif len(input_tensor.shape) > 4:
        raise ValueError(f"Input image has too many dimensions: {input_tensor.shape}")
    output_dict = model(input_tensor)

    #num_detections = int(output_dict['num_detections'])
    #if 'detection_boxes' in output_dict:
    #    boxes = output_dict['detection_boxes']
    return output_dict

def filter_unconfident_predictions(output_dict, threshold=0.5):
    """
    Removes all entries below the confidence threshold from the output dictionary.
    Retains the order of remaining images
    :param output_dict:
    :param threshold:
    :return:
    """
    scores = output_dict['detection_scores']
    save_idxes = np.argwhere(scores >= threshold)
    save_idxes = tf.convert_to_tensor(save_idxes)
    new_scores = tf.gather_nd(scores, indices=save_idxes)
    boxes = output_dict['detection_boxes']
    new_boxes = tf.gather_nd(boxes, indices=save_idxes)
    detection_classes = output_dict['detection_classes']
    new_classes = tf.gather_nd(detection_classes, indices=save_idxes)
    # replace the old with the new
    output_dict['detection_scores'] = new_scores
    output_dict['detection_boxes'] = new_boxes
    output_dict['detection_classes'] = new_classes
    return output_dict

def draw_bounding_boxes(img, output_dict, colors=np.array([[255, 0, 0], [0, 255, 0]]), give_annotated=False):
    boxes = output_dict['detection_boxes']
    img = tf.cast(img, tf.float32)
    img_with_boxes = tf.image.draw_bounding_boxes(img[tf.newaxis, ...], boxes[tf.newaxis, ...], colors)
    converted = tf.cast(img_with_boxes[0], tf.uint8)
    to_show = converted.numpy()

    if not give_annotated:
        cv2.imshow(f"Most Confident bounding boxes", to_show)
    else:
        # returns the np array of the image with bounding boxes already drawn
        return to_show
    #plt.imshow(img_with_boxes[0].numpy().astype(np.int32))
    #plt.show()

def draw_bounding_boxes_with_labels_confidence(img, output_dict, labels_dict, colors=np.array([[255, 0, 0], [0, 255, 0]])):
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']
    classes = output_dict['detection_classes']
    shape = img.shape
    # This is making the assumption that the img is an np.ndarray, which when this is called it *should* be
    # yes I'm aware that I should add error handling, currently it is a todo if I have time later
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for idx, box in enumerate(boxes):
        x = box[1]*shape[1]
        y = box[0]*shape[0]
        # the indices should all match
        class_label_idx = classes[idx].numpy()
        class_label = labels_dict[class_label_idx]
        confidence = scores[idx].numpy()
        # idk how, but somehow mobilenet (the version I'm using) can predict classes not in the coco2017 dataset
        # which is what it's trained on. So, if this ever fails, just default to drawing black text
        try:
            color = tuple(colors[int(class_label_idx)])
        except IndexError:
            color = (0,0,0)
        draw.text((x,y), f"{class_label}: {confidence}", color, font=font)
    cv2.imshow("Labeled Output", np.array(img, dtype=np.uint8))

def main():
    data_dir = "/home/sean/datasets/imagenetv2-top-images-format-val"
    bs = 32
    img_size = (224,224)


    #test_ds = load_dataset(data_dir, bs, img_size)
    #test_ds = test_ds.map(preprocess_input_map_fn)
    model = load_from_hub("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

    #img_batch, label_batch = next(iter(test_ds))
    #img = img_batch[0]
    #print(img)
    """
    sample_img_fname = "./dfd_test_imgs/dfd_map_0.png"
    img = cv2.imread(sample_img_fname)
    input_tensor = preprocess_image(img, img_size)
    output_dict = inference_for_single_image(model, input_tensor)
    output_dict = filter_unconfident_predictions(output_dict, 0.21)
    draw_bounding_boxes(input_tensor, output_dict)
    #output_dict = inference_for_single_image(model, img)
    #output_dict = filter_unconfident_predictions(output_dict, 0.4)
    #draw_bounding_boxes(img, output_dict)
    """

    class_dict = load_pkl_file("./labels/coco2017/labels_dict.pkl")
    colors = load_pkl_file("./labels/coco2017/colors_arr.pkl")
    #cam = NanoCameraCapture(0)
    cam = CameraCapture(0)
    while True:
        #print("start of loop")
        ret, frame = cam.capture_frame_cb()
        #print("captured frame")
        cv2.imshow(f"Camera {cam.idx}", frame)
        as_tensor = preprocess_image(frame)
        output_dict = inference_for_single_image(model, as_tensor)
        #print("got output dict")
        # TODO move to a callback so I can do processing in semi real time and shit don't hang on the main thread
        output_dict = filter_unconfident_predictions(output_dict, 0.4)
        with_boxes = draw_bounding_boxes(frame, output_dict, give_annotated=True, colors=colors)
        draw_bounding_boxes_with_labels_confidence(with_boxes, output_dict, class_dict, colors=colors)
        if cv2.waitKey(1) == 27:
            break
    cam.cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()

