# Utilities for object detector.

import cv2
import numpy as np
import tensorflow as tf

from utils import label_map_util

detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

NUM_CLASSES = 2
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a = b = 0


# Load a frozen inference graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_box_on_image(num_faces_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np, Orientation):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human face (inches) http://www.theaveragebody.com/average_face_size.php
    # added an inch since thumb is not included
    avg_width = 3.0
    # To more easily differentiate distances and detected bboxes

    global a, b
    face_cnt = 0
    color0 = (255, 0, 0)  # red
    color1 = (50, 205, 50)  # green
    for i in range(num_faces_detect):

        if scores[i] > score_thresh:
            if classes[i] == 1:
                class_id = 'Mask'

            if classes[i] == 2:
                class_id = 'No Mask'

            if i == 1:
                color = color1
            else:
                color = color0

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right - left))

            if dist:
                face_cnt = face_cnt + 1
            cv2.rectangle(image_np, p1, p2, color, 3, 1)

            cv2.putText(image_np, 'face ' + str(i) + ': ' + class_id, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])),
                        (int(left), int(top) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
                        (int(im_width * 0.65), int(im_height * 0.9 + 30 * i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

            a = 0  # alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        if face_cnt == 0:
            b = 0
            # print(" no mask")
        else:
            b = 1
            # print(" mask")

    return a, b


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# compute and return the distance from the face to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)
