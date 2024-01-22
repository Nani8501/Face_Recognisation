import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = 'student_images'

images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

encoded_face_train = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}'+"\n")

            
def detect_blink(eye_points, facial_landmarks):
    left_eye = facial_landmarks['left_eye']
    right_eye = facial_landmarks['right_eye']

    left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye, eye_points)
    right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye, eye_points)

    eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

    return eye_aspect_ratio


def calculate_eye_aspect_ratio(eye, eye_points):
    a = np.linalg.norm(eye[eye_points[0]] - eye[eye_points[3]])
    b = np.linalg.norm(eye[eye_points[1]] - eye[eye_points[5]])
    c = np.linalg.norm(eye[eye_points[2]] - eye[eye_points[4]])

    ear = (a + b) / (2.0 * c)
    return ear






EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 3

cap = cv2.VideoCapture(0)
blink_counter = 0
face_detected = False

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS, model='hog')
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

    if len(faces_in_frame) == 0:
        face_detected = False

    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            if not face_detected:
                face_detected = True
                blink_counter = 0

        else:
            face_detected = False
            blink_counter = 0

    if face_detected:
        # Perform eye blink detection only if a face is detected
        landmarks = face_recognition.face_landmarks(imgS, [faceloc])[0]
        for landmark in landmarks:
            left_eye_ear = detect_blink([0, 3, 1, 2, 4, 5], landmark)
            right_eye_ear = detect_blink([0, 3, 1, 2, 4, 5], landmark)

            ear = (left_eye_ear + right_eye_ear) / 2.0

            if ear < EYE_AR_THRESHOLD:
                blink_counter += 1

            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                markAttendance(name)
                blink_counter = 0
                face_detected = False

    cv2.imshow('webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
