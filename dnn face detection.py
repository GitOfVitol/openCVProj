"""
caffe, tensorflow, torch, darknet, DLDT -> 딥러닝 프레임워크(OpenCV dnn 모듈에서 지원)
OpenCV dnn모듈은 C/C++에서도 사용할 수 있어 이식성이 높음
이미 구성된 네트워크에 레이어를 추가하거나 수정하는 기능도 제공함
caffemodel : 딥러닝을 통해 학습된 binary 상태의 모델 파일
prototxt : 해당 신경망 모델의 레이어를 구성하고 속성을 정의

SSD(Single Shot Detector) : 2016년에 발표된 객체 검출 딥러닝 알고리즘
https://taeu.github.io/paper/deeplearning-paper-ssd/ -> 자세한 분석
SSD는 원래 다수의 클래스 객체를 검출할 수 있는데 face detector에 특화됨
OpenCV SSD 구조는 입력층에서 300x300크기의 2차원 BGR 컬러 영상을 사용
이 영상은 Scalar(104, 117, 123)이라는 평균값을 사용하여 정규화 -> train.prototxt에서 확인 가능
출력층 출력 데이터는 추출된 객체의 ID, 신뢰도, 사각형 위치 등의 정보 반환
"""
import cv2
import numpy as np

model_name = './data/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = './data/deploy.prototxt.txt'
min_confidence = 0.3
file_name = './image/marathon_01.jpg'

def detectAndDisplay(frame):
    #caffeframework의 네트워크 모델을 읽어들임. 하나의 Net 객체 반환
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    #caffe framework에서는 원본 이미지가 아니리 binary large object로 변환하여 정보 덩어리를 다룸
    #blob은 n차원 배열이며 caffe는 blob을 이용해 데이터를 저장하고 소통(영상데이터를 NCHW로 표현)
    #N : 영상개수, C : 채널개수, H, W : 영상의 세로랑 가로, 마지막 매개변수는 mean subtraction
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
    #blob을 신경망에 넣기 위해 setinput 함수 사용
    model.setInput(blob)
    #순전파 실행(forward propagation), 반환값은 4차원배열
    detections = model.forward()

    #1x1xNx7의 4차원 배열로 앞에는 1x1은 크게 의미가 없고 N은 가져올 수 있는 최대 box개수(얼굴로 판단되는 객체 후보군), 7은 해당 박스에 대한 데이터
    #7가지 데이터 중 2번은 얼굴 신뢰도고 3~6번은 각각 박스 꼭지점 좌측상단과 우측하단의 x, y좌표로 총 4개
    #이때 좌표 위치는 전체 폭과 높이에 대한 상대적인 위치
    #shape[2]는 N에 해당하며 얼굴 객체 후보들에 대해서 전부 실행한다는 의미
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            #박스를 표현하기 위한 좌표값을 2차원 행렬에 담음
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            print(confidence, startX, startY, endX, endY)

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Face Detection by dnn", frame)


img = cv2.imread(file_name)
(height, width) = img.shape[:2]

detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()