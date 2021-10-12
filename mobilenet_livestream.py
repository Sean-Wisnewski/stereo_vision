import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import cv2

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

def draw_bounding_boxes(img, output_dict, colors=np.array([[255, 0, 0], [0, 255, 0]])):
    boxes = output_dict['detection_boxes']
    img = tf.cast(img, tf.float32)
    img_with_boxes = tf.image.draw_bounding_boxes(img[tf.newaxis, ...], boxes[tf.newaxis, ...], colors)
    converted = tf.cast(img_with_boxes[0], tf.uint8)
    to_show = converted.numpy()

    cv2.imshow(f"Most Confident bounding boxes", to_show)
    #plt.imshow(img_with_boxes[0].numpy().astype(np.int32))
    #plt.show()


def main():
    data_dir = "/home/sean/datasets/imagenetv2-top-images-format-val"
    bs = 32
    img_size = (224,224)


    #test_ds = load_dataset(data_dir, bs, img_size)
    #test_ds = test_ds.map(preprocess_input_map_fn)
    model = load_from_hub("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

    #img_batch, label_batch = next(iter(test_ds))
    #img = img_batch[0]

    #output_dict = inference_for_single_image(model, img)
    #output_dict = filter_unconfident_predictions(output_dict, 0.4)
    #draw_bounding_boxes(img, output_dict)

    cam = CameraCapture(0)
    while True:
        ret, frame = cam.capture_frame_cb()
        cv2.imshow(f"Camera {cam.idx}", frame)
        as_tensor = preprocess_image(frame)
        output_dict = inference_for_single_image(model, as_tensor)
        # TODO move to a callback so I can do processing in semi real time and shit don't hang on the main thread
        output_dict = filter_unconfident_predictions(output_dict, 0.4)
        draw_bounding_boxes(frame, output_dict)
        if cv2.waitKey(1) == 27:
            break
    cam.cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()

