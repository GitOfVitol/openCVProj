import cv2
import face_recognition
import pickle
import time

file_name = './video/son_02.mp4'
encoding_file = 'encodings.pickle'
unknown_name = 'Unknown'
model_method = 'cnn'

def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes =face_recognition.face_locations(rgb, model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        #compare해서 tolerance이상의 값이 나오면 true 출력(미리 임베딩값을 뽑았던 20개의 임베딩값과 각각 비교)
        matches = face_recognition.compare_faces(data["encodings"], encoding, 0.6)
        name = unknown_name

        #true라고 판명된 값들에 대해서만 처리
        if True in matches:
            #matches에서 true인 값의 index만 모음
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            #다른 사람의 임베딩값을 true로 판단할 수도 있으니 true 개수를 count해서 더 큰 값을 선택하기 위한 dictionary 자료형
            counts = {}

            #true로 판명된 임베딩값들의 index를 가지고 for문 진행
            for i in matchedIdxs:
                name = data["names"][i]
                #name값을 key로 집어넣고 반복문 돌면서 +1을해서 count진행
                counts[name] = counts.get(name, 0) + 1

            #key는 해당 함수를 counts dictinary 자료형에 대해서 각각 실행해서 가장 큰 값 가져옴
            #이를 통해 counts에서 가장 많이 count된 key값을 가져올 수 있음
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    # loop over the recognized faces
    # zip은 boxes와 names의 리스트 자료형에 대해서 index가 같은 것들끼리 튜플로된 리스트로 바꿔즘
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if (name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''

        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, line)

    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    # show the output image
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow("Recognition", image)

data = pickle.loads(open(encoding_file, "rb").read())

cap = cv2.VideoCapture(0)
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

cap.release()
cv2.destroyAllWindows()