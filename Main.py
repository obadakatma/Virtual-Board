import os
import time

import cv2 as cv
import numpy as np

import HandTrackingModule as Hand

# reading the header image
headerPath = "Header"
myList = os.listdir(headerPath)
overlayList = []
for imPath in myList:
    img = cv.imread(f'{headerPath}/{imPath}')
    overlayList.append(img)
header = overlayList[0]

# reading the button image
buttonPath = "button pic"
pic = os.listdir(buttonPath)
buttonList = []
for imPath in pic:
    img = cv.imread(f'{buttonPath}/{imPath}')
    buttonList.append(img)
button = buttonList[0]

# declaring variables
drawColor = (255, 255, 255)
brushThikness = 15
eraserThikness = 80
xp, yp = 0, 0
drawPanel = np.zeros((720, 1280, 3), np.uint8)
cTime, pTime = 0, 0

# capturing the video frame
capture = cv.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

# hand detection
detector = Hand.handDetector(detectionCon=0.85)

while True:
    _, frame = capture.read()
    frame = cv.flip(frame, 1)

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    # reading the landmarks of the hand
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("selection mode")
            if y1 < 125:
                if 0 < x1 < 175:
                    # print("pencil")
                    drawColor = (255, 255, 255)
                elif 175 < x1 < 340:
                    print("red")
                    drawColor = (0, 0, 255)
                elif 340 < x1 < 520:
                    # print("green")
                    drawColor = (0, 255, 0)
                elif 520 < x1 < 710:
                    # print("blue")
                    drawColor = (255, 0, 0)
                elif 710 < x1 < 900:
                    # print("orange")
                    drawColor = (0, 165, 255)
                elif 90 < x1 < 1100:
                    # print("purple")
                    drawColor = (255, 0, 255)
                elif 1100 < x1 < 1280:
                    # print("eraser")
                    drawColor = (0, 0, 0)
            elif 320 < y1 < 400:
                if 1200 < x1 < 1280:
                    # print("clear all")
                    drawPanel = np.zeros((720, 1280, 3), np.uint8)
            cv.rectangle(frame, (x1, y1 - 35), (x2, y2 + 35), drawColor, -1)

        if fingers[1] and fingers[2] == False:
            cv.circle(frame, (x1, y1), 12, drawColor, -1)
            # print("drawing   mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv.line(frame, (xp, yp), (x1, y1), drawColor, eraserThikness)
                cv.line(drawPanel, (xp, yp), (x1, y1), drawColor, eraserThikness)
            else:
                cv.line(frame, (xp, yp), (x1, y1), drawColor, brushThikness)
                cv.line(drawPanel, (xp, yp), (x1, y1), drawColor, brushThikness)

            xp, yp = x1, y1
    # merging the main frame with drawing frame
    gray = cv.cvtColor(drawPanel, cv.COLOR_BGR2GRAY)
    _, frameInverse = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
    frameInverse = cv.cvtColor(frameInverse, cv.COLOR_GRAY2BGR)
    frame = cv.bitwise_and(frame, frameInverse)
    frame = cv.bitwise_or(frame, drawPanel)

    # calculating the fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10, 170), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    frame[0:125, 0:1280] = header
    frame[320:400, 1200:1280] = button
    cv.imshow("main", frame)
    # cv.imshow("draw", drawPanel)

    if cv.waitKey(20) & 0xFF == ord(' '):
        break
