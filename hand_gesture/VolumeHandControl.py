import cv2
import mediapipe as mp
import time
import numpy as np
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import HandDetectionModule as htm

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# volume.GetMute()
# volume.GetMasterVolumeLevel()
# volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

wCam, hCam = 640, 480

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)

detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList): 
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255, .5), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255, .5), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255, .5), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0,255),3)

        length = math.hypot(x2-x1, y2-y1)
        if length < 50 : 
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            print(length)

        
    cTime = time.time()

    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)
    
    cv2.imshow('image', img)
    cv2.waitKey(1)