from model.trafficSignsClassification import Model
import pandas as p

# --- labels for the validation data ---
data = p.read_csv("../model/labels.csv", sep=",")
labelNames = data["Name"].tolist()

# --- number of pictures that are shown ---
num = 5

# --- load model and make predictions on the validation data ---
model = Model()
model.evaluateTestData(labelNames, num)
