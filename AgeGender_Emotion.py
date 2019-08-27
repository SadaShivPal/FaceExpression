"""

Face detection

"""

import cv2
import argparse
from time import time
import os
from time import sleep
import numpy as np
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import matplotlib.image as mpimg
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import keras
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort
from keras.preprocessing.image import img_to_array
import sys
import imutils
import dlib
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model
from tkinter import filedialog
from tkinter import *
import multiprocessing
class FaceCV(object):

    """

    Singleton class for face recongnition task

    """

    CASE_PATH = "haarcascade_frontalface_default.xml"

    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"

    #########>>>>>
    model_ethinicity = load_model('keras_FACERACE_trained_model.h5')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=100,desiredLeftEye=(0.32, 0.32))
    array = ["White","Black","Asian","Indian","Other"]
    ##########>>>>>>>>>>>>
    ####>>>>>>
    face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_classifier = load_model('models/_mini_XCEPTION.106-0.65.hdf5', compile=False)
    EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
    
	###########>>
    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):

        if not hasattr(cls, 'instance'):

            cls.instance = super(FaceCV, cls).__new__(cls)

        return cls.instance



    def __init__(self, depth=16, width=8, face_size=64):

        self.face_size = face_size

        self.model = WideResNet(face_size, depth=depth, k=width)()

        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")

        fpath = get_file('weights.18-4.06.hdf5',

                         self.WRN_WEIGHTS_PATH,

                         cache_subdir=model_dir)

        self.model.load_weights(fpath)





    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,

                   font_scale=1, thickness=2):

        size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        x, y = point

        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)

        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)



    def crop_face(self, imgarray, section, margin=40, size=64):

        """

        :param imgarray: full image

        :param section: face detected area (x, y, w, h)

        :param margin: add some margin to the face detected area to include a full head

        :param size: the result image resolution with be (size x size)

        :return: resized image in numpy array with shape (size x size x 3)

        """

        img_h, img_w, _ = imgarray.shape

        if section is None:

            section = [0, 0, img_w, img_h]

        (x, y, w, h) = section

        margin = int(min(w,h) * margin / 100)

        x_a = x - margin

        y_a = y - margin

        x_b = x + w + margin

        y_b = y + h + margin

        if x_a < 0:

            x_b = min(x_b - x_a, img_w-1)

            x_a = 0

        if y_a < 0:

            y_b = min(y_b - y_a, img_h-1)

            y_a = 0

        if x_b > img_w:

            x_a = max(x_a - (x_b - img_w), 0)

            x_b = img_w

        if y_b > img_h:

            y_a = max(y_a - (y_b - img_h), 0)

            y_b = img_h

        cropped = imgarray[y_a: y_b, x_a: x_b]

        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

        resized_img = np.array(resized_img)

        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)



    def detect_face(self):

        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)



        # 0 means the default video capture device in OS

        video_capture = cv2.VideoCapture(0)

        # infinite loop, break by key ESC

        while True:

            if not video_capture.isOpened():

                sleep(5)

            # Capture frame-by-frame

            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ####>>
            rects = self.detector(gray, 2)
            ####~>>>>>>>>>>
            faces = face_cascade.detectMultiScale(

                gray,

                scaleFactor=1.2,

                minNeighbors=10,

                minSize=(self.face_size, self.face_size)

            )

            # placeholder for cropped faces

            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))

            for i, face in enumerate(faces):

                face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)

                (x, y, w, h) = cropped
                ########>>>>>>>>
                for rect in rects:
                    (x, y , w , h) =rect_to_bb(rect)
                    faceAligned = self.fa.align(frame, gray, rect)
                    cv2.imwrite("Img" + str(i) + ".png", faceAligned)
                    faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
                    arr = np.reshape(faceAligned,(1,100,100,3))
                    arr = arr.astype(np.float)
                #############>>>>>>>
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

                face_imgs[i,:,:,:] = face_img
                

            if len(face_imgs) > 0:

                # predict ages and genders of the detected faces

                results = self.model.predict(face_imgs)

                predicted_genders = results[0]

                ages = np.arange(0, 101).reshape(101, 1)

                predicted_ages = results[1].dot(ages).flatten()
                ##########>>>>>>>>>>>>
                pd = self.model_ethinicity.predict(arr) 
                race = self.array[np.asscalar(pd.argmax(axis=1))]
                #########>>>>
                ##########>>>>>
                frontal_faces = sorted( faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = frontal_faces
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi) 
                roi = np.expand_dims(roi, axis=0)
                np.reshape(roi,(48,48,1))
                preds = self.emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                emotion = self.EMOTIONS[preds.argmax()] 
     ########>>>>>>>>
            # draw results

            for i, face in enumerate(faces):

                label = "{}, {} , {},{}".format(emotion,race, int(predicted_ages[i]), "F" if predicted_genders[i][0] > 0.5 else "M") 

                self.draw_label(frame, (face[0], face[1]), label)



            cv2.imshow('Keras Faces', frame)

            if cv2.waitKey(5) == 27:  # ESC key press

                break

        # When everything is done, release the capture

        video_capture.release()

        cv2.destroyAllWindows()





def get_args():

    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "

                                                 "and estimates age and gender for the detected faces.",

                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)



    parser.add_argument("--depth", type=int, default=16,

                        help="depth of network")

    parser.add_argument("--width", type=int, default=8,

                        help="width of network")

    args = parser.parse_args()

    return args



def main():

    args = get_args()

    depth = args.depth

    width = args.width



    face = FaceCV(depth=depth, width=width)



    face.detect_face()



if __name__ == "__main__":

    main()
