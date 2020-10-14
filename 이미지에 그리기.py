#이미지에 그리기
import numpy as np
import cv2

img = cv2.imread("./image/nomadProgramerIcon.png")

#픽셀의 r,g,b값을 가져오는 방법, cv에서는 b, g, r 순서
(b, g, r) = img[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue {}".format(r,g,b))

dot = img[50:100, 50:100]
cv2.imshow("Dot", dot)

#pixel의 R,G,B값 변경
img[50:100, 50:100] = (0,0,255)

#도형그리기 사각형 시작점, 끝점, b/g/r값, 선의 굵기)
cv2.rectangle(img, (150, 50), (200, 100), (0,255,0), 5)

#도형그리기 원 원점의 위치, 반지름, b/g/r값, 선의 두께(-1은 전체가 채워짐)
cv2.circle(img, (275, 75), 25, (0, 255, 255), -1)

#도형그리기 선 시작점, 끝점, b/g/r값, 굵기
cv2.line(img, (350, 100), (400,100), (255, 0, 0), 5)

#text 넣기 출력할 문자, 위치, font, font크기, b/g/r값, font 두께
cv2.putText(img, 'hello', (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
cv2.imshow("Drawing", img)

cv2.waitKey(0)
cv2.destroyAllWindows()