#python drowniness_yawn.py --webcam webcam_index

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
from tb_gateway_mqtt import TBDeviceMqttClient

THINGSBOARD_HOST = 'thingsboard.cs.cf.ac.uk'
ACCESS_TOKEN = 'BRVkeLHpeIsPHPELRCH2'

client = None

def main():
    global client
    client= TBDeviceMqttClient(THINGSBOARD_HOST, username=ACCESS_TOKEN)
    client.connect()
    print("Connected")
    #ip_address = os.popen('''hostname -I''').readline().replace('\n','').replace(',','.')[:-1]
    #mac_address = os.popen('''cat /sys/class/net/*/address''').readline().replace('\n','').replace(',','.')
    #attributes = {
    #    'ip_address': ip_address,
    #    'macaddress': mac_address
    #    }
    #client.send_attributes(attributes)
    #x = 0
    #test = {'x':x}
    #client.send_telemetry(test)
    #time.sleep(1)
    #x = 1
    #test = {'x':x}
    #client.send_telemetry(test)
main()

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def send_data(data):
    ser = serial.Serial('/dev/rfcomm0', baudrate=9600)
    ser.write(data.encode())
    time.sleep(1.0)
    ser.close()


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 15
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
counter_cloud = 0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
vs= VideoStream(usePiCamera=True).start()
print("test")

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("test2")


print("-> Starting Video Stream")
#vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()      #For Raspberry Pi
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                #if alarm_status == False:
                #    alarm_status = True
                #    t = Thread(target=alarm, args=('wake up',))
                #    t.deamon = True
                #    t.start()
                    
                # BLUETOOTH COMMUNCATION
                # E sent to arduino
                data_to_send = 'E'
                send_data(data_to_send)
                
                #counter_cloud += 1
                #print(counter_cloud)

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #if alarm_status2 == False and saying == False:
                #    alarm_status2 = True
                #    t = Thread(target=alarm, args=('get some caffeine in ya',))
                #    t.deamon = True
                #    t.start()
                
                # BLUETOOTH COMMUNCATION
                # Y sent to arduino
                data_to_send = 'Y'
                send_data(data_to_send)
                
                counter_cloud = counter_cloud + 1
                
                print(counter_cloud)
                data ={
                    'counter':counter_cloud
                    }
                client.send_telemetry(data)
                

        else:
            continue
            #alarm_status2 = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()