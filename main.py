from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model import predict_cancer

app=FastAPI()

class InputCan(BaseModel):
    values:List[float]

@app.post('/predict')
def predict(data:InputCan):
    pre=predict_cancer(data.values)
    return {'prediction':pre}
