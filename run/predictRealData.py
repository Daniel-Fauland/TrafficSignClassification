from model.trafficSignsClassification import Model
import pandas as p

# --- labels for the validation data ---
data = p.read_csv("../model/labels.csv", sep=",")
labelNames = data["Name"].tolist()

# --- location of the real data folder ---
imagePath = "../realData"

# --- load model and make predictions on the validation data ---
model = Model()
model.evaluateRealData(labelNames, imagePath)
