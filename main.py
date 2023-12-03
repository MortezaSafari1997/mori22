from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load the saved logistic regression model
logistic_regression_model = pickle.load(open('model.sav', 'rb'))

# Define the request body model
class InputData(BaseModel):
    Age: float
    Sex: float
    ALB: float
    ALP: float
    ALT: float
    AST: float
    BIL: float
    CHE: float
    CHOL: float
    CREA: float
    GGT: float
    PROT: float

# Endpoint for Logistic Regression model
@app.post("/predict/hepatitis")
def predict_hepatitis(data: InputData):
    input_data = pd.DataFrame([data.dict()]).to_numpy()
    prediction = logistic_regression_model.predict(input_data)[0]
    
    # Assuming 1 corresponds to hepatitis positive, and 0 corresponds to negative
    if prediction == 1:
        result = "Negative"
    else:
        result = "Positive"
    
    return {"prediction": result}


