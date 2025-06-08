import time
import datetime
import pandas as pd
import json
import datetime
import pickle
import os

from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel


# Suppress warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PARENT_DIR, "data"))


loaded_model = None

app = FastAPI()


class PredictionInput(BaseModel):
    data: list


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Application startup")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Please ensure the data directory is available.")

    print("Load the model from a pickle file")

    model_path = os.path.join(DATA_DIR, "regression_model.pkl")

    try:
        with open(model_path, "rb") as model_file:
            loaded_model = pickle.load(model_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is available.") 

    print("Model loaded and cached at startup")

    yield

    print("Application shutdown")

# Assign the lifespan context manager to the app
app.router.lifespan_context = lifespan

@app.get("/")
def doc():
    return {
        "message": "Welcome to the model service",
        "documentation": "If you want to see the OpenAPI specification, navigate to the /redoc/ path on this server."
    }



@app.post("/predict")
async def predict(
    input_data: PredictionInput,
    debug: bool = False):
    
    global loaded_model
    
    if loaded_model is None:
        raise RuntimeError("Model is not loaded. Please ensure the application has started correctly.")
    
    start_time = time.time()

    try:

        prediction = loaded_model.predict([input_data.data])[0]

    except Exception as e:
        print(f"Failed to run model eferience with {e} error")
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

