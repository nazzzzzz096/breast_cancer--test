import numpy as np
import pickle

def load_model():
    with open("model.pkl")as f:
        model=pickle.load(f)

    return model

def predict_cancer(sample):
    sample=np.array(sample).reshape(1,-1)
    model=load_model()
    return int(model.predict(sample)[0])