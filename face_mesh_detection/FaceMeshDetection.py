import cv2
import mediapipe as mp

import time

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

mpFaceMeshDetection = mp.solutions.face_mesh
faceMeshDetection = mpFaceMeshDetection.FaceMesh(max_num_faces=2)

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMeshDetection.process(imgRGB)
    # print(results)
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLm, mpFaceMeshDetection.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLm.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)
                
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)