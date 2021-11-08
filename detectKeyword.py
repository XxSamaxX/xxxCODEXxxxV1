# This is the third script called predict.py in there, we predict the keyword that is said in the microphone
#
# Steps to take:
# 1. Libraries
# 2. Load model
# 3. Predict the keyword

# Step 1
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model


# Step 2
def getSpectrogram(keyword, n):
    y, sr = librosa.load("./wavsamples/" + keyword + n + ".wav")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.figure(figsize=(1.28, 1.28))
    plt.axis('off')
    plt.axes([0., 0., 1., 1., ], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig("./spectrograms/" + keyword + n + ".png", bbox_inches=None, pad_inches=0)
    plt.close()


# Step 3
def predict(model):
    predict_dir = "./spectrograms"
    predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    predict_generator = predict_datagen.flow_from_directory(predict_dir,
                                                            target_size=(128, 128),
                                                            batch_size=128,
                                                            class_mode='binary')

    predictions = model.predict_generator(predict_generator, steps=1)
    print(predictions)


# Step 4
if __name__ == "__main__":
    model = load_model("model.h5")
    predict(model)