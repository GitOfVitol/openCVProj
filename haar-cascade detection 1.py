"""
haar-cascade detection

object detection 하는 방법으로 빠르다는 장점이 있지만 정확도가 CNN보다 좋지 않고 예외 상황에 약함
미리 정해진 방식에 의해 인식을 하기 때문에 정확도 낮을 수 밖에 없음, kernel을 사용하는 점은 CNN과 유사하기도 함

cascade classifier(다단계 분류)를 이용한 객체 검출
다수의 객체 이미지와 객체가 아닌 이미지를 cascade 함수로 트레이닝 시켜 객체 검출을 하는 머신러닝 기반 접근법
edge, line, four-rectangle feature들을 활용
"""

import cv2
import numpy as np

def detectAndDisplay(frame):
    #channel이 많으면 정확도가 떨어질 수 있어서 gray로 scale을 하고 진행
    #equalizeHist : 인식하기 편하게 단순화시키기 위해 사용, histogram에서 하나로 집중된 픽셀 값을 골고루 분포시킴
    #numpy를 활용한 구현도 가능하면 numpy는 컬러 이미지에 적용가능
    #equalizeHist는 grayscale 이미지만 인자로 받고 리턴값도 grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    #-- Detect faces
    #detectMultiScale(image, scaleFactor, minNeighbors, flags, minSize, maxSize)
    #scaleFactor는 scale pyramid 만들 때 얼마나 줄어들게 할건지 - 1이상이고 1.03이면 3%가 줄어든다는 의미
    #minNeighbors는 얼굴 사이의 최소 간격 값이 크면 덜 검출되지면 정확도가 오름
    #얼굴의 위치를 리스트로 리턴 (x, y, w, h)와 같은 투플 형식, (x, y)는 검출된 얼굴의 최상단 위치, w랑 h는 가로, 세로크기
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        cv2.imshow('Capture - Face detection', frame)


img = cv2.imread("face.jpg")
resized = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
(height, width) = resized.shape[:2]

#미리 학습된 data를 불러옴
face_cascade_name = './data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

#face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_alt.xml')
#이런식으로 한 번에 써도 됨
face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

#Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

detectAndDisplay(resized)

cv2.waitKey(0)
cv2.destroyAllWindows()