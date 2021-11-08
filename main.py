# ### JARVIS PROJECT
# ### Created by XxSamaxX and co-piloted by GPT-3 with engine davinci-codex <3
# ##
# # This script is the root of the project, the main functions should be here well defined and simplified,
# a lot of work to do Dx
#
# top libraries

import cv2
import pyaudio
import threading
import concurrent.futures
import sys
from time import sleep

# assets
import GenerativePretrainedTransformers as gpt
import detectFaces as Df
import SpeechToText as Stt
import HandTracking as Htm
import predictKeyword

# Load trained model faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# For hands (for now only 1 user)
detector = Htm.handDetector(maxHands=2)

# Open camera
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


# Create typing text to print
def typing(text):
    for char in text:
        sleep(0.04)
        sys.stdout.write(char)
        sys.stdout.flush()


# Parallel process with typing and text_to_speech
def parallel(text):
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_tasks = {executor.submit(Stt.text_to_speech, text), executor.submit(typing, text)}
        for future in concurrent.futures.as_completed(future_tasks):
            try:
                data = future.result()
            except Exception as e:
                print(e)


# Create threadable process
def multiThread_with_TTS(text):
    threading.Thread(
        target=parallel, args=(text,), daemon=None
    ).start()


def thread_in_background(function=parallel, args=None):
    thread = threading.Thread(target=function, args=args, daemon=None)
    thread.start()


def readfile(path='./data/predictKEYWORD.txt'):
    f = open(path, 'r')
    content = f.read()
    f.close()
    return int(content)


if __name__ == "__main__":
    predictKeyword.writefile()
    parallel("\n [LOAD DONE]")
    while True:
        ret, img = cam.read()
        imgH = detector.findHands(img)
        lmList, bbox = detector.findPosition(imgH )
        fingers = detector.fingersUp
        ids, confidences = Df.detect(img, prediction=70)
        if threading.active_count() == 1:
            thread_in_background(predictKeyword.autoPredict, ("./wavsamples/sampleWord.wav",
                                                            2,
                                                            pyaudio.paInt16,
                                                            1,
                                                            44100,
                                                            1024))
        if len(confidences) > 0 and (readfile() > 34):
            print(" [GPT-3]")
            parallel(" Can I help you?")
            print("\n [{0}] ".format(ids[0]))
            textUser = Stt.STT_google()
            if textUser:
                for char in textUser:
                    sleep(0.04)
                    sys.stdout.write(char)
                    sys.stdout.flush()
            parallel(gpt.requests(textUser, ids[0]))
            predictKeyword.writefile()
        cv2.imshow('Jarvis', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27 and id != "":  # Press 'ESC' for exiting video
            print(" [System]")
            parallel(" Bye " + ids[0] + ", see you later!")
            break
        if k == 32 and id != "":
            print(" [System]")
            multiThread_with_TTS(" You are " + ids[0] + ", hi!")
        if k == 32 and id != "":
            print(" [System]")
            multiThread_with_TTS(" You are " + ids[0] + ", hi again!")

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
