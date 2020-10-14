#이미지 채널 조작
import cv2
import numpy as np

img = cv2.imread("./image/food.jpg")
resized = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

(height, width) = resized.shape[:2]
center = (width//2, height//2)

#각각 b/g/r에 대한 filter를 씌운 이미지, gray scale로 표현됨
(Blue, Green, Red) = cv2.split(resized)
"""
cv2.imshow("Red", Red)
cv2.imshow("Green", Green)
cv2.imshow("Blue", Blue)
"""

"""
gray scale이 아니라 우리한테 익숙한 형태로 하기 위해 해당 값 이외는 전부 0이미지로
zeros = np.zeros(resized.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, Red]))
cv2.imshow("Green", cv2.merge([zeros, Green, zeros]))
cv2.imshow("Blue", cv2.merge([Blue, zeros, zeros]))
"""

#gray filter
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray filter", gray)
#사람이 생각하는 filter
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV filter", hsv)
#사람이 보는 직관적인 filter
lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB filter", lab)

#앞에서 split 한걸 다시 merge하면 원래 사진으로 나옴
BGR = cv2.merge([Blue, Green, Red])
cv2.imshow("Merge", BGR)

cv2.waitKey(0)
cv2.destroyAllWindows()