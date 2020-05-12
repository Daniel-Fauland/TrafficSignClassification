import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split  # function to split the data very easy


class Model():
    def __init__(self, parameters=None):
        # --- prevent TF from using more VRAM than the GPU actually has ---
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # --- Do not use unless you have problems running TF2 with gpu ---
        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force CPU Usage, instead of GPU

        # --- parameters ---
        self.path = "../trainingData"
        self.labels = "labels.csv"
        if parameters is not None:
            self.batchSize = parameters["batchSize"]
            self.epochs = parameters["epochs"]
            self.validation = parameters["validation"]
        else:
            self.validation = 0.2


    # =================================
    def loadData(self):
        # --- read images and labels ---
        count = 0
        images, categories = [], []
        data = os.listdir(self.path)

        for x in range(len(data)):
            folder = os.listdir(self.path + "/" + str(count))
            for file in folder:
                singleImage = cv2.imread(self.path + "/" + str(count) + "/" + file)
                images.append(singleImage)
                categories.append(count)
            count += 1
            print("Categorie " + str(count) + " done.")

        # --- split trainingData into train and validation ---
        x_train, x_val, y_train, y_val = train_test_split(images, categories, test_size=self.validation)
        print("loading data done.")
        return x_train, x_val, y_train, y_val


    # =================================
    def preprocessData(self, x_train, x_val, y_train, y_val):
        # --- normalize images ---
        def normalize(img):
            img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)  # Grayscale image
            # img = cv2.equalizeHist(np.uint8(img))                    # Optimize Lightning
            img = img / 255.0                                        # Normalize px values between 0 and 1
            return img

        for x in range(len(x_train)):
            x_train[x] = normalize(x_train[x])

        for x in range(len(x_val)):
            x_val[x] = normalize(x_val[x])

        # --- transform the data to be accepted by the model ---
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        x_train = np.array(x_train)
        x_val = np.array(x_val)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
        print("preprocessing data done.")
        return x_train, x_val, y_train, y_val


    # =================================
    def model(self, x_train, x_val, y_train, y_val):
        # --- build the model based on a CNN ---
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(43))

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batchSize,
                            validation_data=(x_val, y_val))
        model.save("../model/savedModel.h5")
        print("training model done.")
        return history, model, x_val, y_val


    # =================================
    def results(self, history, model, x_val, y_val):
        # --- plot a graph showing the accuracy over the epochs
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()


    # =================================
    def loadModel(self, x_val):
        # --- load the model using the load_model function from keras ---
        model = keras.models.load_model("../model/savedModel.h5")
        prediction = model.predict(x_val)
        print("loading model done.")
        return prediction


    # =================================
    def trainModel(self):
        # --- loads data, prepossess data, trains model, plots results ---
        print("initializing...")
        x_train, x_val, y_train, y_val = self.loadData()
        x_train, x_val, y_train, y_val = self.preprocessData(x_train, x_val, y_train, y_val)
        history, model, x_val, y_val = self.model(x_train, x_val, y_train, y_val)
        self.results(history, model, x_val, y_val)
        print("\n=================================")
        print("The highest tested accuracy on the validation data is: " +
              str(round(max(history.history["val_accuracy"]) * 100, 2)) + "%")
        print("Number of epochs: " +
              str(history.history["val_accuracy"].index(max(history.history["val_accuracy"])) + 1))
        print("=================================")


    # =================================
    def evaluateTestData(self, labelNames, num):
        # --- loads data, prepossess data, loads model, makes predictions on validation data ---
        print("initializing...")
        x_train, x_val, y_train, y_val = self.loadData()
        x_train, x_val, y_train, y_val = self.preprocessData(x_train, x_val, y_train, y_val)
        prediction = self.loadModel(x_val)

        # --- iterate over a set number of images to predict them ---
        for i in range(num):
            img = np.array(x_val[i]*255, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.xlabel("Actual: " + labelNames[y_val[i]])
            plt.title("Prediction: " + labelNames[np.argmax(prediction[i])])
            plt.show()


    # =================================
    def showImage(self, img):
        # --- this function shows any image that gets passed to it ---
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
