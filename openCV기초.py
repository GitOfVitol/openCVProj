#기초
import numpy as np
import cv2

print(cv2.__version__)

img = cv2.imread("./image/food.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("food", img)
cv2.imshow("food - gray", gray)

cv2.waitKey(0)
cv2.destroyWindow()

#보간법으로 픽셀 변경
resized = cv2.resize(img, None, fx = 0.2, fy = 0.2, interpolation=cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)