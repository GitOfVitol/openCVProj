#이미지 픽셀좌표 체계
import numpy as np
import cv2

img1 = cv2.imread("./image/food.jpg")
print("width: {} pixels".format(img1.shape[1]))
print("height: {} pixels".format(img1.shape[0]))
print("channels: {} pixels".format(img1.shape[2]))

img2 = cv2.imread("./image/nomadProgramerIcon.png")
print("width: {} pixels".format(img2.shape[1]))
print("height: {} pixels".format(img2.shape[0]))
print("channels: {} pixels".format(img2.shape[2]))

cv2.imshow("food", img1)
cv2.imshow("nomad", img2)
cv2.waitKey(0)
cv2.imwrite("food.png", img1)
cv2.destroyAllWindow()