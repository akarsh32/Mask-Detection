import argparse
import datetime
import threading
import time

import cv2
import numpy as np
from imutils.video import VideoStream

from utils import detector_utils as detector_utils


class MaskDetection:
    def __init__(self):
        self.lst1 = []
        self.lst2 = []
        self.vs = None
        self.outputFrame = None
        self.lock = threading.Lock()
        # vs = VideoStream(usePiCamera=1).start()
        # vs = cv2.VideoCapture('rtsp://192.168.1.64')
        self.vs = VideoStream(0).start()
        time.sleep(1.0)

        self.detection_graph, self.sess = detector_utils.load_inference_graph()

    def __count_no_of_times(self, lst):
        x = y = cnt = 0
        for i in lst:
            x = y
            y = i
            if x == 0 and y == 1:
                cnt = cnt + 1
        return cnt

    def __detect_mask(self):
        # Detection confidence threshold to draw bounding box
        score_thresh = 0.80

        # Orientation of machine
        Orientation = 'bt'
        # input("Enter the orientation of face progression ~ lr,rl,bt,tb :")

        # For Machine
        Line_Perc1 = float(15)

        # For Safety
        Line_Perc2 = float(30)

        # max number of faces we want to detect/track
        num_faces_detect = 10

        # Used to calculate fps
        start_time = datetime.datetime.now()
        num_frames = 0

        im_height, im_width = (None, None)

        try:
            frame = self.vs.read()
            frame = np.array(frame)

            if im_height is None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(frame, self.detection_graph, self.sess)

            # Draw bounding boxes and text
            a, b = detector_utils.draw_box_on_image(num_faces_detect, score_thresh, scores, boxes, classes,
                                                    im_width,
                                                    im_height, frame, Orientation)
            self.lst1.append(a)
            self.lst2.append(b)
            no_of_time_face_detected = no_of_time_face_crossed = 0
            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)

            # acquire the lock, set the output frame, and release the lock
            with self.lock:
                 self.outputFrame = frame.copy()

            no_of_time_face_detected = self.__count_no_of_times(self.lst2)
            no_of_time_face_crossed = self.__count_no_of_times(self.lst1)

        except KeyboardInterrupt:
            no_of_time_face_detected = self.__count_no_of_times(self.lst2)
            no_of_time_face_crossed = self.__count_no_of_times(self.lst1)

    def generate(self):
        # loop over frames from the output stream
        while True:
            self.__detect_mask()
            # wait until the lock is acquired
            with self.lock:
                # check if the output frame is available, otherwise skip the iteration of the loop
                if self.outputFrame is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
            # yield the output frame in the byte format
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
