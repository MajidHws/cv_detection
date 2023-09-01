import cv2
import mediapipe as mp
import time

import HandDetectionModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=True)
    if len(lmList):
        print(lmList[4])
    cTime = time.time()

    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)
    
    cv2.imshow('image', img)
    cv2.waitKey(1)