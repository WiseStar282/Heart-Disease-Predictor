import numpy as np

def predict(model, new_data):
    return model.predict(new_data)

def predict_probabilities(model, new_data):
    return model.predict_proba(new_data)
