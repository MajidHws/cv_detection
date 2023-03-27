import cv2
import mediapipe as mp
import time


class FaceDetector():

    def __init__(self, minDetectorCon=0.5):

        self.minDetectorCon = minDetectorCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):

        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        # print(results)
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                # mpDraw.draw_detection(img, detection)
                # print(id, detection)
                # print(id, detection.score)
                # print(id, detection.location_data.relative_bounding_box)

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin *ih), int(bboxC.width * iw), int(bboxC.height *ih)
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                bboxs.append([bbox, detection.score])
        return img, bboxs


    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):

        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        #TOP LEFT X,Y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)

        #TOP RIGHT X1,Y
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        # #BOTTOM LEFT X, Y1
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)

        # #BOTTOM RIGHT X1,Y1
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

        return img
        
        
        


def main():

    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = FaceDetector()
    
    while True:
        success, img = cap.read()
        
        img, bboxs =detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()