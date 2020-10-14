"""
darknet이라는 framework 사용 -> linux에서만 사용 가능
tensorflow를 darknet에 적용한 것이 darkflow(사용이 복잡함)
openCV에서 3.4.2 정도 이후의 버전에서 YOLO사용 가능
"""

import cv2
import numpy as np

min_confidence = 0.1

#미리 학습된 파일을 load, weight는 훈련된 모델, cfg는 알고리즘 관한 설정
net = cv2.dnn.readNet("./data/yolov3.weights", "./data/yolov3.cfg")

#object들을 80개 정도 class로 구분 가능하며 그 파일을 load해야 함
#파일을 열어서 한줄씩 calsses 배열에 넣음
classes = []

#with open(파일경로, 모드) as 파일객체: -> close 없어도 with as 구문을 빠져나가면 자동으로 close()
with open("./data/coco.names", "r") as f:
    #strip은 문자열의 양쪽 끝 공백과 \n을 삭제(중간에 있는 건 x)
    #readlines를 하면 한 줄이 각각의 리스트 원소로 들어감
    classes = [line.strip() for line in f.readlines()]

#네트워크의 마지막 레이어로 인식되는 연결되어 있찌 않은 출력 레이어의 이름
output_layers = net.getUnconnectedOutLayersNames()
#물건마다 랜덤하게 다른 색을 줌, random 중 균등분포 이용, size는 classes x 3 -> channel(r,g,b)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread("./image/yolo_01.jpg")
height, width, channels = img.shape

#yolo는 320, 609, 416 size의 blob을 accept할 수 있음(작을수록 빠르지만 정확도는 낮음)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

#각각의 output layer에 대한 detection에 대해서 confidence, class id, bounding box 정보를 가져옴
for out in outs:
    for detection in out:
        scores = detection[5:]
        #argmax는 가장 큰 값의 index반환
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#Non-max suppresion을 통해서 노이즈 제거, 0.4는 NMS을 사용할 때의 threshold
#여러개의 겹쳐있는 박스 중 confidence가 가장 높은 박스 선택, indexes는 confidence가 가장 높은 상자의 index
indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} {confidences[i]*100:.2f}%"
        print(i, label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), font, 1, (0, 255, 0), 1)
print(classes)
cv2.imshow("YOLO Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()