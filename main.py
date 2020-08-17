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

    count = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = imutils.grab_contours(count)
    count = sorted(count, key=cv2.contourArea, reverse=True)

    puzzleCount = None

    for obj in count:
        peri = cv2.arcLength(obj, True)
        approximation = cv2.approxPolyDP(obj, 0.02 * peri, True)

        if len(approximation) == 4:
            puzzleCount = approximation
            break

    if puzzleCount is None:
        raise Exception(("Couldn't find any Sudoku outline"))

    puzzle = four_point_transform(imgUMat, puzzleCount.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCount.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle Threshold", threshold)
        # cv2.waitKey(0)

        output = imgUMat.copy()
        cv2.drawContours(output, [puzzleCount], -1, (0, 255, 0), 2)
        cv2.imshow('Puzzle Contours', output)
        # cv2.waitKey(0)

        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    return (puzzle, warped)


find(test, True)
