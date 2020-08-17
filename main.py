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

    contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)

    puzzleContour = None

    for obj in contour:
        peri = cv2.arcLength(obj, True)
        approximation = cv2.approxPolyDP(obj, 0.02 * peri, True)

        if len(approximation) == 4:
            puzzleContour = approximation
            break

    if puzzleContour is None:
        raise Exception(("Couldn't find any Sudoku outline"))

    puzzle = four_point_transform(imgUMat, puzzleContour.reshape(4, 2))
    warped = four_point_transform(gray, puzzleContour.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle Threshold", threshold)
        # cv2.waitKey(0)

        output = imgUMat.copy()
        cv2.drawContours(output, [puzzleContour], -1, (0, 255, 0), 2)
        cv2.imshow('Puzzle Contours', output)
        # cv2.waitKey(0)

        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    return (puzzle, warped)


def extractionDigit(cell, debug=False):
    threshold = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    threshold = clear_border(threshold)

    contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    if len(contour) == 0:
        return None

    contours = max(contour, key=cv2.contourArea)
    mask = np.zeros(threshold.shape, dtype='uint8')
    cv2.drawContours(mask, [contours], -1, 255, -1)

    h, w = threshold.shape
    percentFilled = cv2.countNonZero(mask) / float(h * w)

    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(threshold, threshold, mask=mask)

    if debug:
        cv2.imshow("Cell Threshold", threshold)
        cv2.imshow('Digit', threshold)
        cv2.waitKey(0)

    return digit


find(test, True)
