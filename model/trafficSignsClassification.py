# --- Import libraries ---
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU Usage, instead of GPU
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split  # Function to split data very easy
import json

# --- Prevent TF from using more VRAM than the GPU actually has ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# --- Parameters ---
path = "../data"
labels = "labels.csv"
batchSize = 50
epochs = 10
testing = 0.2  # 20% of training data
validation = 0.2  # 20% of training data


# --- Prepare data ---
# - Read images and labels -
count = 0
images, categories = [], []
data = os.listdir(path)

for x in range(len(data)):
    folder = os.listdir(path + "/" + str(count))
    for file in folder:
        singleImage = cv2.imread(path + "/" + str(count) + "/" + file)
        images.append(singleImage)
        categories.append(count)
    print("Categorie " + str(count) + " done.")
    count += 1


# - Split data into train, test, validation -
x_train, x_test, y_train, y_test = train_test_split(images, categories, test_size=testing)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation)


# --- Preprocess data ---
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Grayscale image
    img = cv2.equalizeHist(img)                  # Optimize Lightning
    img = img / 255.0                            # Normalize px values between 0 and 1
    return img


for x in range(len(x_train)):
    x_train[x] = preprocess(x_train[x])

for x in range(len(x_test)):
    x_test[x] = preprocess(x_test[x])

for x in range(len(x_val)):
    x_val[x] = preprocess(x_val[x])


# --- Transform data to be accepted by the model
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)
x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)


# --- model ---
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(43))


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])



history = model.fit(x_train, y_train, epochs=10, batch_size=batchSize, validation_data=(x_val, y_val))



# --- Plot a graph showing the accuracy over the epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(x_val,  y_val, verbose=2)
print(test_acc)



