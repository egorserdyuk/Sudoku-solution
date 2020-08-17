from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

test = "assets/example.jpg"


def find(image, debug=False):
    imgUMat = cv2.imread(image)
    gray = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshold = cv2.bitwise_not(threshold)

    if debug:
        cv2.imshow('Puzzle', threshold)
        cv2.waitKey(0)


find(test, True)
