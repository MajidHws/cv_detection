import cv2
import mediapipe as mp
import time


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)

    cv2.imshow('Imaeg', img)
    cv2.waitKey(1)