from model.trafficSignsClassification import trainModel

# --- Change some parameters here ---
parameters = {"batchSize":50, "epochs":10, "validation":0.2}

# --- trains the network
model = trainModel(parameters)
model.run()

