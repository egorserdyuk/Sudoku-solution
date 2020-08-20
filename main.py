from recognizer import extractionDigit, find
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import imutils
import cv2

print(r"Load classifier")
model = load_model('assets/model.h5')

print(r"Processing")
image = cv2.imread('assets/example.jpg')
image = imutils.resize(image, width=600)

(puzzleImage, warped) = find(image, debug=False)

board = np.zeros((9, 9), dtype='int')

X = warped.shape[1]
Y = warped.shape[0]

cellLocs = []

for y in range(0, 9):
    row = []

    for x in range(0, 9):
        startX = x * X
        startY = y * Y

        endX = (x + 1) * X
        endY = (y + 1) * Y

        row.append((startX, startY, endX, endY))

        cell = warped[startY:endY, startX:endX]
        digit = extractionDigit(cell, debug=True)

        if digit is not None:
            plc = cv2.resize(digit, (28, 28))
            plc = plc.astype('float') / 255.0
            plc = img_to_array(plc)
            plc = np.expand_dims(plc, axis=0)

            pred = model.predict(plc).argmax(axis=1)[0]
            board[y, x] = pred

    cellLocs.append(row)

print(r"Board")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

print("[INFO] solving Sudoku puzzle...")
solution = puzzle.solve()
solution.show_full()

for (cellRow, boardRow) in zip(cellLocs, solution.board):
    for (box, digit) in zip(cellRow, boardRow):
        startX, startY, endX, endY = box

        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY

        cv2.putText(puzzleImage, str(digit), (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)
