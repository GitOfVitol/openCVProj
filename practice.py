import cv2
import numpy as np

model_name = './data/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = './data/deploy.prototxt.txt'
min_confidence = 0.3
file_name = './image/yejimong.jpg'

def detectAndDisplay(frame):
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence*100)
            textY = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, text, (startX, textY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("DNN - Detection", frame)

image = cv2.imread(file_name)
couple = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
(height, width) = couple.shape[:2]

detectAndDisplay(couple)

cv2.waitKey(0)
cv2.destroyAllWindows()