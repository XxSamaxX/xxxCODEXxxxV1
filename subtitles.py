import os
import sys
import moviepy.editor as mp
import speech_recognition as sr
import datetime


#Crea una script que dado un video .mkv este lo convierta a .wav
def convert_to_wav(video_name, audio_name):
    try:
        pathin = "S:\\IA\\PycharmProjects\\pythonProject\\data"+"\\"+video_name
        pathout = "S:\\IA\\PycharmProjects\\pythonProject\\data"+"\\"+audio_name
        video = mp.VideoFileClip(pathin)
        audio = video.audio
        # create a new audio file
        audio.write_audiofile(pathout+".wav")
        return True
    except:
        print("Not found or bad format in video")
        return False


# Ahora crea una funcion que separe un .wav en chunks de 5 segundos
def split_wav(audio_name):
    path = "S:\\IA\\PycharmProjects\\pythonProject\\data"+"\\"+audio_name
    # comprueba que el archivo existe y cuantos segundos tiene
    if os.path.isfile(path):
        audio = mp.AudioFileClip(path)
        duration = audio.duration
        # si el audio es menor que 5 segundos no lo separa
        if duration < 5:
            print("El audio es menor que 5 segundos")
        else:
            path = "S:\\IA\\PycharmProjects\\pythonProject\\data\\spilts"+"\\"+audio_name
            #si el audio es mayor que 5 segundos lo separa en chunks de 5 segundos en un bucle
            for i in range(0, int(duration), 5):
                # file name as 000001.wav | 000002.wav | etc
                filename = str(i).zfill(6) + ".wav"
                # create a new audio file
                audio.subclip(i, i + 5).write_audiofile(path[:-4] + filename)


    else:
        print("El archivo no existe")

def traduction(path="./wavsamples/Morgan", language="en-EN"):
    textReturn = ""
    r = sr.Recognizer()
    num = 0
    #for each file in the directory add 10 sec in a variable
    # var= 00:00:00,000 --> 00:00:10,000
    # variable in next iteration is 00:00:10,000 --> 00:00:20,000
    # etc
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            num += 1
            # if first time
            if num == 1:
                # create a variable to save the first time
                start = "00:00:00,000"
                end = "00:00:05,000"
            else:
                # num is the file number
                # for var in num multiply by 5 to get the format HH:MM:SS,000
                sec = str(num * 5)
                # fomat the variable to HH:MM:SS,000 with datetime
                var = datetime.timedelta(seconds=int(sec))# var: H:MM:SS
                # edit the variable to the format HH:MM:SS,000
                time = str(var)
                # convert to string
                end = "0" + time + ",000"
            with sr.AudioFile(path+"/"+filename) as source:
                audio = r.record(source)
            try:
                text = r.recognize_google(audio, language=language)
                print(text)
                with open(path+"/"+"list.srt", "w") as f:
                    # Save write to return later
                    # textReturn = textReturn + "wavs/" + filename + " | " + text + "\n"
                    if num == 1:
                        textReturn = textReturn + str(num) + "\n" + start + " --> " + end + "\n" + text
                    else:
                        textReturn = textReturn + "\n\n" + str(num) + "\n" + start + " --> " + end + "\n" + text
                    f.write(textReturn)
                    start = end
            except:
                print("Error")
    return textReturn


if __name__ == "__main__":
    video_name = "video.mkv"
    audio_name = "audio.wav"
    #if convert_to_wav(video_name, "audio"):
    #    split_wav(audio_name)
    traduction("./data/spilts", "es-ES")
