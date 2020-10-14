#이미지 변형하기
import cv2
import numpy as np

img = cv2.imread("./image/food.jpg")

#resize(img, dsize, fx, fy, interpolation)
#dsize는 manual로 지정, fx/fy는 가로 세로 사이즈의 배수, interpolation은 보간법
#size를 줄일때는 inter_area, 키울때는 inter_cubic, inter_linear
#resized = cv2.resize(img, (np.int64(img_width*0.2), np.int64(img_height*0.2)), interpolation=cv2.INTER_AREA_ 도 가능
resized = cv2.resize(img, None, fx = 0.2, fy = 0.2, interpolation=cv2.INTER_AREA)
(height, width) = resized.shape[:2]

#//는 floor division -> 7.5면 7로 버림
center = (width // 2, height // 2)

#warpAffine(img, M, (w,h)) -> img : source, M : 2x3 변환 matrix, (w, h) : 출력 이미지 사이즈
# [1, 0, 100], [0, 1, 100] -> x 방향으로 100, y 방향으로 100pixel 이동, 양수는 오른쪽 아래
trans_mat = np.float32([[1, 0, 100], [0, 1, 100]])
translation = cv2.warpAffine(resized, trans_mat, (width, height))
cv2.imshow("Moved down: +, up: - and right: +, left -", translation)

#getRoatationMatrix2D(center, angle, scale) -> 변환행렬 생성함수
#center는 image 중심점, angle은 회전 방향(양수는 시계 반대 방향), scale은 출력 크기
rotation_mat = cv2.getRotationMatrix2D(center, 180, 1.0)
rotation = cv2.warpAffine(resized, rotation_mat, (width, height))
cv2.imshow("Rotation", rotation)

#horizontal : 1, vertical : 0, both : -1
flipped = cv2.flip(resized, -1)
cv2.imshow("flipped", flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()