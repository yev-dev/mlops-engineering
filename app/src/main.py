import time
from contextlib import asynccontextmanager

from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import pandas as pd
import json
import datetime
from app.models import univariate, multivariate, single_timeseries, multi_timeseries

app = FastAPI()



@app.get("/")
def doc():
    return {
        "message": "Welcome to the model service",
        "documentation": "If you want to see the OpenAPI specification, navigate to the /redoc/ path on this server."
    }


class PredictionInput(BaseModel):
    data: list


@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    debug: bool = False):
    
    start_time = time.time()
    try:

        data = pd.DataFrame(
            columns=[
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
            ],
            data=[data],
        )

        prediction = loaded_model.predict(data)

    except Exception as e:
        raise e
    finally:
        elapsed_time = time.time() - start_time

        results = {
            "prediction": prediction.tolist(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if debug:
            print(f"Prediction took {elapsed_time:.2f} seconds")
            results.update({"debug_elapsed_time": elapsed_time})
            
    return results



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

