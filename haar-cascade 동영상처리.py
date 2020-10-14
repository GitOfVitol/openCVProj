import cv2
import numpy as np

face_cascade_name = './data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
file_name = './video/obama_01.mp4'

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
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


face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# -- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

# -- 2. Read the video stream
#실시간으로 하려면 webcam이나 라즈베리파이 카메라 활용 가능
#VideoCapture(0) 이런식으로 하면 웹캠 사용 가능 -> 장치에 달린 카메라 활용 가능
cap = cv2.VideoCapture(file_name)

#cap이 초기화되지 않은 경우 error를 리턴하므로 이 메소드를 통해 확인 가능
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

#라이브로 들어오는 비디오를 프레임별로 캠쳐해야하므로 무한루프 사용
while True:
    #한 frame씩 재생되는 비디오를 읽음 -> 제대로 읽으면 ret이 true 아니면 false 값이 됨
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cap 객체를 반드시 해제하자.
cap.release()
cv2.destroyAllWindows()