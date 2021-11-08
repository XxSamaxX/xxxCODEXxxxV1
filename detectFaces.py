import cv2
import sys
import threading
import concurrent.futures
import HandTracking as htm
import SpeechToText as stt
import GenerativePretrainedTransformers as gpt3
from time import sleep

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# For hands
detector = htm.handDetector(maxHands=2)

# initiate id counter
id = 0
# Some flags
flag = False
gesFlag = False
# initiate timer
timeFlip = 0
# initiate string
stringText = ""


# names related to ids: example ==> Marcelo: id=1,  etc
# Load names from recog folder
def namesFaces(path="./"):
    # Open the file
    file = open(path, "r")
    # Read the file
    names = file.read()
    # Split the names
    names = names.split("\n")
    # Close the file
    file.close()
    # Return the vector
    return names


names = namesFaces("./face_recog/names.txt")
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
# this function is used to convert the given string to speech


def typing(text):
    for char in text:
        sleep(0.04)
        sys.stdout.write(char)
        sys.stdout.flush()


def parallel(text):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_tasks = {executor.submit(stt.text_to_speech, text), executor.submit(typing, text)}
        for future in concurrent.futures.as_completed(future_tasks):
            try:
                data = future.result()
            except Exception as e:
                print(e)


def multiThread_with_TTS(text):
    # running = threading.Event()
    # running.set()

    # thread = threading.Thread(target=parallel, args=(text,))
    # thread.start()
    # running.clear()
    # thread.join()
    threading.Thread(
        target=parallel, args=(text,), daemon=None
    ).start()


def multiThread_with_HANDS(img):
    threading.Thread(
        target=detector.findHands, args=(img,), daemon=True
    ).start()
    lmList, bbox = detector.findPosition(img)
    threading.Thread(
        target=detector.findPosition, args=(img,), daemon=True
    ).start()
    if detector.fingersUp is not None:
        fingers = detector.fingersUp
        #print(detector.fingersUp)


def gestures_Suport(fingersup, id="unknown"):
    rock = [1, 0, 1, 0, 0, 1]
    rockFlip = [0, 0, 0, 1, 1, 0]
    peace = [1, 0, 1, 1, 0, 0]
    talk = [1, 0, 0, 0, 0, 1]
    if (fingersup == rock) | (fingersup == rockFlip):
        multiThread_with_TTS(" Rock and Roll!")
    if fingersup == peace:
        multiThread_with_TTS(" Peace!")
    if fingersup == talk:
        text = stt.STT_google()
        print("\n ["+id+"] "+text)
        gpt3.requests(text, id)


def detect(img, prediction=100):
    # img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    ids = []
    confidences = []
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if confidence < prediction:
            text = names[id]
            ids.append(text)
            text = round(100 - confidence)
            confidences.append(text)
            cv2.rectangle(img, (x, y), (x + w, y + h), (38, 102, 0), 2)

        else:
            text = "unknown"
            ids.append(text)
            text = round(100 - confidence)
            confidences.append(text)
            cv2.rectangle(img, (x, y), (x + w, y + h), (178, 0, 0), 2)
    return ids, confidences


if __name__ == "__main__":
    while True:
        ret, img = cam.read()
        imgH = detector.findHands(img)
        lmList, bbox = detector.findPosition(imgH)
        fingers = detector.fingersUp
        if fingers is not None:
            if timeFlip % 11 == 9:
                flag = False
            if not flag:
                flag = True
                gestures_Suport(fingers, id)
        # multiThread_with_HANDS(img)
        img = cv2.flip(img, 1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                if timeFlip == 0:
                    timeFlip = 1
                    if stringText != id:
                        stringText = id
                        multiThread_with_TTS(" Hi " + id + ".")
                        # parallel(" Hi " + id)
                else:
                    timeFlip += 1
                    if timeFlip > 35:
                        timeFlip = 0
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27 and id != "":  # Press 'ESC' for exiting video
            parallel(" Bye " + id + ", see you later!")
            break
        if k == 32 and stringText != id:
            multiThread_with_TTS(" You are " + id + ", hi!")
            # parallel(" You are " + id+", hi!")
        if k == 32 and stringText == id:
            multiThread_with_TTS(" You are " + id + ", hi again!")
            # parallel(" You are " + id + ", hi again!")

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()