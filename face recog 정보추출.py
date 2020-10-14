import cv2
import face_recognition
import pickle

dataset_paths = ['./dataset/son/', './dataset/tedy/']
names = ['Son', 'Tedy']
number_images = 10
image_type = '.jpg'
encoding_file = 'encodings.pickle'
#cnn or hog 둘 다 가능, CNN이 더 정확하지만 느림
model_method = 'cnn'

#임베딩 값을 얻어서 넣어줄 배열과 해당 임베딩 값이 누구의 임베딩인지를 알려주는 배열 선언
knownEncodings = []
knownNames = []

#enumerate로 자동으로 index부여 하면 son의 dataset이 0번째 tedy가 1번째이고 각각에 대해 for문 돌림
for (i, dataset_path) in enumerate(dataset_paths):
    name = names[i]

    #각 dataset에 10개의 이미지가 있고 해당 이미지들 하나씩 for문 돌려서 임베딩값 가져옴(128x128)
    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type

        image = cv2.imread(file_name)
        #openCV는 bgr로 돼있으니까 RGB로 바꾸는 필터 사용
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #가져온 사진에서 face의 위치를 가져옴, model은 cnn 사용
        boxes = face_recognition.face_locations(rgb, model=model_method)

        #임베딩값 추출(encoding)
        encodings = face_recognition.face_encodings(rgb, boxes)

        print(file_name, name, encodings[0])
        knownEncodings.append(encodings[0])
        knownNames.append(name)

data = {"encodings": knownEncodings, "names": knownNames}
#wb는 binary 쓰기모드
f = open(encoding_file, "wb")
#객체 계층 구조를 직렬화
f.write(pickle.dumps(data))
f.close()