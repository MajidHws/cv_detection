import cv2
import time
import PoseDetectionModule as pm
cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)
    if len(lmList):
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (0,0,0), cv2.FILLED)
    # print(lmList)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)