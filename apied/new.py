import cv2
import sys


image = sys.argv[1]
img = cv2.imread(image)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('thumb',gray)
cv2.waitKey(0)
