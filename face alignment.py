import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

predictor_file = './data/shape_predictor_68_face_landmarks.dat'
image_file = './image/tedy.jpg'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (300, 300)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

image = cv2.imread(image_file)
image = cv2.resize(image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
image_origin = image.copy()

(image_height, image_width) = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top()
    endY = rect.bottom()
    return (startX, endX, startY, endY)

for (i, rect) in enumerate(rects):
    #openCV와 dlib에서 사용하는 좌표가 달라서 바꾸기위해 만든 함수
    (x, y, w, h) = getFaceDimension(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
    show_parts = points[EYES]

    right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
    left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")

    cv2.circle(image, (right_eye_center[0,0], right_eye_center[0,1]), 5, (0, 0, 255), -1)
    cv2.circle(image, (left_eye_center[0,0], left_eye_center[0,1]), 5, (0, 0, 255), -1)

    #나중에 기울어진 값만큼 사진을 돌리기 위해서 left/right 눈의 center좌표와 또 하나의 점으로 직각삼각형 만듦
    #그 직각삼각형의 마지막 점은 left의 x좌표와 right의 y좌표와 같음
    cv2.circle(image, (left_eye_center[0, 0], right_eye_center[0, 1]), 5, (0, 255, 0), -1)
    cv2.line(image, (right_eye_center[0, 0], right_eye_center[0, 1]), (left_eye_center[0, 0], left_eye_center[0, 1]), (0, 255, 0), 2)
    cv2.line(image, (right_eye_center[0,0], right_eye_center[0,1]), (left_eye_center[0,0], right_eye_center[0,1]), (0, 255, 0), 1)
    cv2.line(image, (left_eye_center[0,0], right_eye_center[0,1]), (left_eye_center[0,0], left_eye_center[0,1]), (0, 255, 0), 1)

    eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
    eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
    degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180

    eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
    aligned_eye_distance = left_eye_center[0, 0] - right_eye_center[0, 0]
    #기울어진 빗변을 내렸을 때의 비율을 구해서 나중에 wapaffine
    scale = aligned_eye_distance / eye_distance

    #eyes의 중간점을 찾아서 이를 기준으로 돌리기 위해 좌표 필요
    eyes_center = ((left_eye_center[0, 0] + right_eye_center[0, 0]) // 2,
                   (left_eye_center[0, 1] + right_eye_center[0, 1]) // 2)
    cv2.circle(image, eyes_center, 5, (255, 0, 0), -1)

    #기준점인 eye_center를 가지고 앞에서 계산한 degree만큼 scale을 해서 돌리기 위해 필요한 matrix
    metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)
    cv2.putText(image, "{:.5f}".format(degree), (right_eye_center[0, 0], right_eye_center[0, 1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
                            flags=cv2.INTER_CUBIC)

    cv2.imshow("warpAffine", warped)
    (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)
    croped = warped[startY:endY, startX:endX]
    output = cv2.resize(croped, OUTPUT_SIZE)
    cv2.imshow("output", output)

    for (i, point) in enumerate(show_parts):
        x = point[0, 0]
        y = point[0, 1]
        cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

cv2.imshow("Face Alignment", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
