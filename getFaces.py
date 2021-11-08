# the following script is written in python language, in which a script is created that calls the web camera of the ordered to detect faces and be able to recognice faces individually, check first if that face is in face_recog folder, if exist write a text said "hi, <user id>", if not exist create a folder and save images inside face_recog (Ex:"face_recog/samu/User.samu.1.jpg"). Cascade path is "haarcascades/", recognition faces path is  "face_recog/" (Edited by Sama, thanks Codex)

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
flag = False
for file in os.listdir("face_recog/"):
            if file.endswith(".jpg"):
                if file.startswith("User."+str(face_id)):
                    if not flag:
                        flag = True
                        count = int(file.split(".")[-2])
                    if flag and count < int(file.split(".")[-2]):
                        count = int(file.split(".")[-2])
if not flag:
    count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        print(count)
        print("\n")
        # Save the captured image into the datasets folder
        cv2.imwrite("face_recog/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 300: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()