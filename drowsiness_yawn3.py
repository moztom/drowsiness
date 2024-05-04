import argparse
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import os
import serial
import paho.mqtt.client as mqtt
import json

# Thingsboard configuration

THINGSBOARD_HOST = 'thingsboard.cs.cf.ac.uk'
ACCESS_TOKEN = 'BRVkeLHpeIsPHPELRCH2'

data = {
    "count":0
    }

count=0

# Sending data to Thingsboard
def send_data(data):
    ser = serial.Serial('/dev/rfcomm0', baudrate=9600)
    ser.write(data.encode())
    time.sleep(1.0)
    ser.close()

# Updating Threshold values via Thingsboard
# data from Thingsboard
thresholds = {
        'YAWN_THRESH': 15,
        'EYE_THRESH': 0.26,
        'EYE_CLOSED_FRAMES': 30
    }

client = mqtt.Client()

def on_message(client, userdata, msg):
    data1 = json.loads(msg.payload)
    print('data1')
    if data1['method'] == 'setThreshold':
        set_threshold(data1['params'],msg)
        print('set_threshold called')
    
def set_threshold(params,msg):
    global thresholds
    if 'YAWN_THRESH_INC' in params:
        new_yawn_thresh = thresholds['YAWN_THRESH'] + 1
        thresholds['YAWN_THRESH'] = new_yawn_thresh
    elif 'YAWN_THRESH_DEC' in params:
        new_yawn_thresh = thresholds['YAWN_THRESH'] - 1
        thresholds['YAWN_THRESH'] = new_yawn_thresh
    elif 'EYE_THRESH_INC' in params:
        new_eye_thresh = thresholds['EYE_THRESH'] + 0.01
        thresholds['EYE_THRESH'] = new_eye_thresh
    elif 'EYE_THRESH_DEC' in params:
        new_eye_thresh = thresholds['EYE_THRESH'] - 0.01
        thresholds['EYE_THRESH'] = new_eye_thresh
    elif 'EYE_FRAMES_INC' in params:
        new_eye_frames = thresholds['EYE_CLOSED_FRAMES'] + 1
        thresholds['EYE_CLOSED_FRAMES'] = new_eye_frames
    elif 'EYE_FRAMES_DEC' in params:
        new_eye_frames = thresholds['EYE_CLOSED_FRAMES'] - 1
        thresholds['EYE_CLOSED_FRAMES'] = new_eye_frames
    print(f'Updated thresholds: {thresholds}')
    client.publish(msg.topic.replace('request', 'response'), json.dumps({'status':'success'}),1)

client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883, 60)
client.loop_start()
client.subscribe('v1/devices/me/rpc/request/+')
client.on_message = on_message

# Calculate the eye aspect ratio (ear)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

# Calculate eye aspect ratios (ear) for the left and right eyes
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    ear = (left_ear + right_ear) / 2.0
    return (ear, left_eye, right_eye)

# Calculate the distance between the lips
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

print("Loading model")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("Starting video")
vs= VideoStream(usePiCamera=True).start()

time.sleep(1.0)

COUNTER = 0

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        left_eye = eye[1]
        right_eye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if (ear < thresholds['EYE_THRESH']):
            COUNTER += 1

            if (COUNTER >= thresholds['EYE_CLOSED_FRAMES']):
                    
                # BLUETOOTH COMMUNCATION
                # 'E' sent to arduino
                data_to_send = 'E'
                send_data(data_to_send)
                
                # Increment drowsiness count and update Thingsboard
                count+=1
                
                data = {
                    "count":count
                    }
                
                client.publish('v1/devices/me/telemetry', json.dumps(data), 1)

        else:
            COUNTER = 0

        if (distance > thresholds['YAWN_THRESH']):
                
                # BLUETOOTH COMMUNCATION
                # 'Y' sent to arduino
                data_to_send = 'Y'
                send_data(data_to_send)
                
                # Increment drowsiness count and update Thingsboard
                count+=1
                
                data = {
                    "count":count
                    }
                
                client.publish('v1/devices/me/telemetry', json.dumps(data), 1)

        else:
            continue

    cv2.imshow("Frame", frame)
    
    #Keyboard Interupt
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

client.loop_stop()
client.disconnect()
cv2.destroyAllWindows()
vs.stop()