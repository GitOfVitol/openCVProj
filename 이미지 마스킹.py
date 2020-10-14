#이미지 마스킹
import cv2
import numpy as np

img = cv2.imread("./image/food.jpg")
img_height, img_width = img.shape[:2]

resized = cv2.resize(img,(np.int64(img_width*0.2), np.int64(img_height*0.2)), interpolation=cv2.INTER_AREA)
resized_height, resized_width = resized.shape[:2]
center = (resized_width//2, resized_height//2)

#numpy를 이용해 width, height 부분을 전부 0으로 채움
mask = np.zeros(resized.shape[:2], dtype= "uint8")
cv2.circle(mask, center, 300, (255, 255, 255), -1)

#resized 전체 이미지에 대해서 bitwise_and 연산 진행
#mask에서 검정색이 아닌 경우 bitwise 연산을 하면 1이 되므로 검정색이 아닌 부분만 남음
masked = cv2.bitwise_and(resized, resized, mask=mask)
cv2.imshow("mask", masked)

cv2.waitKey(0)
cv2.destroyWindow()