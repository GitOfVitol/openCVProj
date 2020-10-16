import cv2
import dlib
import numpy as np

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = './data/shape_predictor_68_face_landmarks.dat'
image_file = './image/marathon_02.jpg'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#1은 얼굴인식할 때 layer개수
rects = detector(gray, 1)


for (i, rect) in enumerate(rects):
    #학습된 predictor파일에 이미지와 얼굴 detect한 것을 넣고 68개의 point의 x, y좌표를 배열에 넣어줌
    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[ALL]
    for (i, point) in enumerate(show_parts):
        x = point[0,0]
        y = point[0,1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(image, "{}".format(i + 1), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

cv2.imshow("Face Landmark", image)
cv2.waitKey(0)