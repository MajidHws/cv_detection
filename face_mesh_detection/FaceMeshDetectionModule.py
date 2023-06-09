import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):


        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMeshDetection = mp.solutions.face_mesh
        self.faceMeshDetection = self.mpFaceMeshDetection.FaceMesh(max_num_faces=2)

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMeshDetection.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLm in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLm, self.mpFaceMeshDetection.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                    face = []
                    for id, lm in enumerate(faceLm.landmark):
                        ih, iw, ic = img.shape
                        x, y = int(lm.x*iw), int(lm.y*ih)
                        # print(([x,y]))
                        cv2.putText(img, str('id'), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
                        face.append([x,y])
                    faces.append(face)

        return img, faces

def main():

    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = FaceMeshDetector()
    
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, True)
        # if len(faces):
        #     print(len(faces))

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()