# USAGE
# python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt --input example_activities.mp4
# python human_activity_reco.py --model resnet-34_kinetics.onnx --classes action_recognition_kinetics.txt

import numpy as np
import argparse
import imutils
import sys
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

def load_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('model.h5')
    return model

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="", help="optional path to video file")
args = vars(ap.parse_args())

# load the contents of the class labels file, then define the sample
# duration (i.e., # of frames for classification) and sample size
# (i.e., the spatial dimensions of the frame)
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
while True:
    frames = []

    for i in range(0, SAMPLE_DURATION):
        (grabbed, frame) = vs.read()

        if not grabbed:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)

        # otherwise, the frame was read so resize it and add it to our frames list
        frame = imutils.resize(frame, width=600)
        frames.append(frame)

    # now that our frames array is filled we can construct our blob
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # pass the blob through the network to obtain our human activity recognition predictions
    net.setInput(blob)
    outputs = net.forward()
    label = CLASSES[np.argmax(outputs)]

    for frame in frames:
        # draw the predicted activity on the frame
        model = load_model()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, 'Activity: ' + label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            left_up_cord_x = x
            left_up_cord_y = y
            right_bottom_cord_x = x + w  # may say width of rectangle
            right_bottom_cord_y = y + h  # may say height of rectangle
            cv2.rectangle(frame, (left_up_cord_x, left_up_cord_y), (right_bottom_cord_x, right_bottom_cord_y), color, stroke)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            cv2.putText(frame, 'Emotion: ' + emotion_dict[max_index], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255),
                        2, cv2.LINE_AA)

        cv2.imshow("Activity Recognition", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break