import argparse
import os
from time import time

import align.detect_face as detect_face
import cv2
import numpy as np
import tensorflow as tf
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils

logger = Logger()

detectEmotion = True
def main():
    global colours, img_size
    args = parse_args()
    videos_dir = args.videos_dir
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold

    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))

            minsize = 40  # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor

            for filename in os.listdir(videos_dir):
                logger.info('All files:{}'.format(filename))
            for filename in os.listdir(videos_dir):
                suffix = filename.split('.')[1]
                if suffix != 'mp4' and suffix != 'avi':  # you can specify more video formats if you need
                    continue
                video_name = os.path.join(videos_dir, filename)
                directoryname = os.path.join(output_path, filename.split('.')[0])
                logger.info('Video_name:{}'.format(video_name))
                #cam = cv2.VideoCapture(video_name)
                cam = cv2.VideoCapture(0)
                c = 0
                while True:
                    final_faces = []
                    addtional_attribute_list = []
                    ret, frame = cam.read()
                    if not ret:
                        logger.warning("ret false")
                        break
                    if frame is None:
                        logger.warning("frame drop")
                        break

                    frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    print('shape of gray')
                    print(gray.shape)
                    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if c % detect_interval == 0:
                        img_size = np.asarray(frame.shape)[0:2]
                        mtcnn_starttime = time()
                        faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                                                                factor)
                        logger.info("MTCNN detect face cost time : {} s".format(
                            round(time() - mtcnn_starttime, 3)))  # mtcnn detect ,slow
                        face_sums = faces.shape[0]
                        if face_sums > 0:
                            face_list = []
                            for i, item in enumerate(faces):
                                score = round(faces[i, 4], 6)
                                if score > face_score_threshold:
                                    det = np.squeeze(faces[i, 0:4])

                                    # face rectangle
                                    det[0] = np.maximum(det[0] - margin, 0)
                                    det[1] = np.maximum(det[1] - margin, 0)
                                    det[2] = np.minimum(det[2] + margin, img_size[1])
                                    det[3] = np.minimum(det[3] + margin, img_size[0])
                                    face_list.append(item)

                                    # face cropped
                                    bb = np.array(det, dtype=np.int32)

                                    # use 5 face landmarks  to judge the face is front or side
                                    squeeze_points = np.squeeze(points[:, i])
                                    tolist = squeeze_points.tolist()
                                    facial_landmarks = []
                                    for j in range(5):
                                        item = [tolist[j], tolist[(j + 5)]]
                                        facial_landmarks.append(item)
                                    if args.face_landmarks:
                                        for (x, y) in facial_landmarks:
                                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                                    dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                        np.array(facial_landmarks))

                                    # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                    item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                                    addtional_attribute_list.append(item_list)

                            final_faces = np.array(face_list)
                            emotion = 'Happy'
                    face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    emotion_classifier = load_model('models/_mini_XCEPTION.106-0.65.hdf5', compile=False)
                    EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
                    
                    frontal_faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

                    if len(frontal_faces) > 0:
                        frontal_faces = sorted(frontal_faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                        (fX, fY, fW, fH) = frontal_faces
                        roi = gray[fY:fY + fH, fX:fX + fW]
                        roi = cv2.resize(roi, (48, 48))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi) 
                        roi = np.expand_dims(roi, axis=0)
                        np.reshape(roi,(48,48,1))
                        print(roi.shape)
                        preds = emotion_classifier.predict(roi)[0]
                        emotion_probability = np.max(preds)
                        label = EMOTIONS[preds.argmax()] 

                    trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, detect_interval)

                    c += 1

                    emoTracker = ''
                    print(trackers)
                    for d in trackers:
                        if not no_display:
                            d = d.astype(np.int32)
                            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                            
                            if final_faces != []:
                                print('ID %d Detect' %(d[4]))
                                

                                if label != emoTracker:
                                    emoTracker = label
                                cv2.putText(frame, 'ID : %d  DETECT, EMOTION : %s' % ((d[4]),emoTracker),(d[0] - 10, d[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4,
                                            colours[d[4] % 32, :] * 255, 2)
                                
                            else:
                                cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.75,
                                            colours[d[4] % 32, :] * 255, 2)

                    if not no_display:
                        frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='videos')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
