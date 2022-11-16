import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Image Attendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findencodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttandance(name):
    with open('Attandance D/At1.csv', 'r+') as f:
        myDatalist = f.readline()
        nameList =[]
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

encodeListknown = findencodings(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        matchs = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matchs[matchIndex]:
#<0.50:
            name = classNames[matchIndex].upper()
#markAttandance(name)
#else: name = 'Unknown'
        y1, x2, y2, x1 = faceLoc #photo box
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4 #photo same size
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) #border color
        cv2.rectangle(img, (x1, y2-35), (x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        markAttandance(name)
    cv2.imshow('KSRCT Hostel Webcam', img)
    cv2.waitKey(1)
