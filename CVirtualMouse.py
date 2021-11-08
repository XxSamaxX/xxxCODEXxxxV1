import cv2
import numpy as np
import HandTracking as htm
import mouse
import autopy

#################
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
#################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
flag = False
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    print(detector.fingersUp)

    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[8][1:]

        fingers = detector.fingersUp
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        if fingers[1] == 1:
            if fingers[2] == 1 and flag == False:
                flag = True
            x3 = np.interp(x2, (frameR, wCam-frameR), (0,wScr))
            y3 = np.interp(y2, (frameR, hCam-frameR), (0,hScr))

            clocX = plocX +(x3 -plocX) /smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr-clocX,clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[2] == 0 and flag:
            print(lmList)
            mouse.click()
            flag=False

        if fingers[0]==0 and flag==True:
            mouse.click()
            mouse.right_click()
            flag=False


    cv2.imshow("Image", img)
    cv2.waitKey(1)

