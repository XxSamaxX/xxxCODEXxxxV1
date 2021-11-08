import os
import sys
import time

import speech_recognition as sr
import pyttsx3 # TAL VEZ SE PUEDA ENTRENAR TOKEN VOICES EN PYTTSX3
from gtts import gTTS



def STT_google():
    try:
        time.time()
        r = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            audio = r.listen(source)
        text = r.recognize_google(audio, language="es-ES")
        return text
    except Exception as e:
        #print(" Can you repeat it?...")
        text_to_speech("Can you repeat it?...")
        return None


def text_to_speech(text, language="en"):
    engine = pyttsx3.init()
    engine.setProperty('rate', 220)
    engine.say(text)
    engine.runAndWait()
    del engine
    # speach = gTTS(text, lang=language, slow=False)
    # speach.save("./data/gpt3VOICE.mp3")
    # os.system("start ./data/gpt3VOICE.mp3 tempo 1.9")



def waitingkey(id, keyword="exit"):
    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        audio = r.listen(source)
    try:
        value = r.recognize_google(audio, language="es-ES")
        print(" ["+id+"] {}".format(value))
        if value == keyword:
            return True
    except sr.UnknownValueError:
        print("\nNone")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
    return False


def traduction(path="./wavsamples/Morgan"):
    textReturn = ""
    r = sr.Recognizer()
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            print(filename)
            with sr.AudioFile(path+"/"+filename) as source:
                audio = r.record(source)
            try:
                text = r.recognize_google(audio, language="en-EN")
                print(text)
                with open(path+"/"+"list.txt", "w") as f:
                    # Save write to return later
                    textReturn = textReturn + "wavs/" + filename + " | " + text + "\n"
                    f.write(textReturn)
            except:
                print("Error")
    return textReturn


if __name__ == "__main__":
    if len(sys.argv) > 1:
        traduction(sys.argv[1])
    else:
        traduction()
    #print(STT_google())