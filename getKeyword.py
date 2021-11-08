# the following script is written in python language, in which a script is created that calls the microphone constantly to create chunks of 2 seconds and obtain the spectrogram of them, then a model is created to train the set of keywords obtained previously and finally the model is used to predict the keyword that is said in the microphone.
#
# This is the first script called getWordsWAV.py in there, we obtain first the .wav from microphone (50 sample from default, 2 seconds for each sample) Path folder for .wav is /wavsKEYWORD
# Example: ./wavsKEYWORD/sampleWord0.wav    ./wavsKEYWORD/sampleWord1.wav    ...    ./wavsKEYWORD/sampleWord49.wav
#
# Steps to take:
# 1. Libraries
# 2. Open microphone and get audios
#                 * when 2 sec is down wait to press a key from keyboard to continue
#                 * save samples in ./wavsamples
# 3. Create a loop to take 50 samples

# Step 1
import os
import wave
import pyaudio


# Step 2
def getWordWAV(keyword, n):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FOLDER = "./wavsamples"
    WAVE_OUTPUT_FILENAME = keyword + n + ".wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(os.path.join(WAVE_OUTPUT_FOLDER, WAVE_OUTPUT_FILENAME), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# Step 3
if __name__ == "__main__":
    for i in range(0, 50 ):
        print("Press any key to continue [{0}/50]".format(i))
        input()
        getWordWAV("GPT-3", str(i))