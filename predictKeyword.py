# This is the third script called predictKeyword.py in there, we predict with microphone if we say the weyword or not, for this we use the model from folder(./model/)
# Example: ./model/keras_metadata.pb    ./model/saved_model.pb     ./model/variables/     ./model/assets
#
# Steps to take:
# 1. Libraries
# 2. Open microphone and streaming
# 3. Get and load the model
# 4. Predict the keyword
# 5. Close microphone and streaming

# Step 1
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
import pyaudio
import wave

# Step 2
# Get and load the model
model = tf.keras.models.load_model("./model/")
# Step 3
# Open microphone and streaming
#FORMAT = pyaudio.paInt16
#CHANNELS = 1
#RATE = 44100
#CHUNK = 1024
#RECORD_SECONDS = 2
#WAVE_OUTPUT_FILENAME = "./wavsamples/sampleWord.wav"
#audio = pyaudio.PyAudio()

# start Recording
#stream = audio.open(format=FORMAT, channels=CHANNELS,
#                    rate=RATE, input=True,
#                    frames_per_buffer=CHUNK)
#print("recording...")
#frames = []

# Step 4
# Predict the keyword
#for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#    data = stream.read(CHUNK)
#    frames.append(data)
#print("finished recording")
# stop Recording
#stream.stop_stream()
#stream.close()
#audio.terminate()

#waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#waveFile.setnchannels(CHANNELS)
#waveFile.setsampwidth(audio.get_sample_size(FORMAT))
#waveFile.setframerate(RATE)
#waveFile.writeframes(b''.join(frames))
#waveFile.close()

# Step 5
# Close microphone and streaming

# Main
# Predict the keyword


def obtainKeyword(path="./wavsamples/sampleWord.wav", record=2, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)
    frames = []
    for i in range(0, int(rate / chunk * record)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    waveFile = wave.open(path, 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(format))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()



def predictKeyword():
    y, sr = librosa.load("./wavsamples/sampleWord.wav")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.figure(figsize=(1.28, 1.28))
    plt.axis('off')
    plt.axes([0., 0., 1., 1., ], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig("./spectrograms/sampleWord.png", bbox_inches=None, pad_inches=0)
    plt.close()
    img = plt.imread("./spectrograms/sampleWord.png")
    img = img.reshape(1, 256, 256, 1)
    prediction = model.predict(img)
    return prediction


def getAcuracy(prediction):
    max = np.argmax(prediction)
    return max


def writefile(path='./data/predictKEYWORD.txt', content="0"):
    f=open(path, 'w')
    f.write(content)
    f.close()


def autoPredict(path="./wavsamples/sampleWord.wav", record=2, format=pyaudio.paInt16, channels=1, rate=44100, chunk=1024):
    obtainKeyword(path, record, format, channels, rate, chunk)
    acuracy = getAcuracy(predictKeyword())
    #print(acuracy)
    writefile(content=str(acuracy))
    return acuracy

# Predict the keyword
#prediction = predictKeyword()

# Get the index of the maximum value
#index = np.argmax(prediction)
#print(index)