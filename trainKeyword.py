# This is the second script called trainModel.py in there, we train first the model from .wav files in the sample folder(./wavsKEYWORD/)
# Example: ./wavsKEYWORD/sampleWord0.wav    ./wavsKEYWORD/sampleWord1.wav    ...    ./wavsKEYWORD/sampleWord49.wav
#
# Steps to take:
# 1. Libraries
# 2. Obtain spectrograms from .wav files
# 3. Create a model
# 4. Train the model

# Step 1
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf


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
# Step 3
# Create a model
# We use the function tf.keras.Sequential to create the model.
# Input: None
# Output: model
def createModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(256, 256, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(50, activation='softmax'))
    model.summary()
    return model


# Step 4
# Train the model
# We use the function model.fit to train the model.
# Input: model, training data, labels, epochs, batch size, validation data, validation labels
# Output: None
def trainModel(model, x_train, y_train, epochs, batch_size, x_test, y_test):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))


# Step 5
# Save the model
# We use the function model.save() to save the model.
# Input: model, path
# Output: None
def saveModel(model, path):
    model.save(path)


# Main
# We create the model
model = createModel()

# We train the model
# Obtain spectrograms from .wav files
for i in range(50):
    getSpectrogram("GPT-3", str(i))

# Save it as npy files in folder. Ex: "./spectrograms/npy/sampleWord.png.npy"
def saveSpectrogram(keyword):
    for i in range(50):
        img = plt.imread("./spectrograms/" + keyword + str(i) + ".png")
        img = img.reshape(1, 256, 256, 1)
        np.save("./spectrograms/npy/" + keyword + str(i) + ".png.npy", img)


saveSpectrogram("GPT-3")
# Load the spectrograms
x_train = []
y_train = []
for i in range(40):
    x_train.append(np.load("./spectrograms/npy/GPT-3" + str(i) + ".png.npy"))
    y_train.append(i)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.reshape(40, 256, 256, 1)

# Load the validation data
x_test = []
y_test = []
for i in range(10):
    x_test.append(np.load("./spectrograms/npy/GPT-3" + str(i + 40) + ".png.npy"))
    y_test.append(i + 40)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape(10, 256, 256, 1)

# Train the model
trainModel(model, x_train, y_train, 1000, 40, x_test, y_test)

# Save the model
saveModel(model, "./model")