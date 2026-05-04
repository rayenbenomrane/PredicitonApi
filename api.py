from fastapi import FastAPI
import pandas as pd
import pickle
from attrition1 import predict_employee # pyright: ignore[reportMissingImports]
app = FastAPI()

# Load trained model
model = pickle.load(open("attrition_model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Attrition Prediction API is running"}

@app.post("/predict")
def predict(employee: dict):
    return predict_employee(employee)