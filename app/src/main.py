import time
import datetime
import pandas as pd
import json
import datetime
import pickle
import os
import psutil

from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel

#Monitoring integration
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator


# Suppress warnings 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PARENT_DIR, "data"))


# Create Prometheus metrics
cpu_usage_gauge = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_usage_gauge = Gauge("memory_usage_bytes", "Memory usage in bytes")
prediction_gauge = Gauge("model_prediction", "Model prediction value")
model_latency = Gauge("model_latency_seconds", "Time taken for model prediction")

# Counters
request_counter = Counter("total_requests", "Total number of prediction requests")
success_counter = Counter(
    "successful_predictions", "Total number of successful predictions"
)
failure_counter = Counter("failed_predictions", "Total number of failed predictions")

# Histograms for latency
latency_histogram = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10),  # Customize buckets as needed
)

# Histogram for prediction values
prediction_histogram = Histogram(
    "prediction_values", "Distribution of prediction values"
)


app = FastAPI(debug=True)

# Expose default metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Expose custom metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class PredictionInput(BaseModel):
    data: list


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):

    global loaded_model
    """Lifespan context manager to handle application startup and shutdown."""

    print("Application startup")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}. Please ensure the data directory is available.")

    print("Loading the model from a pickle file")

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


@app.middleware("http")
async def add_process_metrics(request: Request, call_next):
    response = await call_next(request)
    cpu_usage_gauge.set(psutil.cpu_percent())
    memory_usage_gauge.set(psutil.virtual_memory().used)
    return response

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
    
    if loaded_model is None:
        raise RuntimeError("Model is not loaded. Please ensure the application has started correctly.")
    
    request_counter.inc()
    if debug:
        print(f"Received input data: {input_data.data}")
    if not isinstance(input_data.data, list):
        raise ValueError("Input data must be a list.")
    
    start_time = time.time()

    try:

        prediction_value = loaded_model.predict([input_data.data])[0]
        prediction_gauge.set(prediction_value)
        prediction_histogram.observe(prediction_value)

        success_counter.inc()

    except Exception as e:
        print(f"Failed to run model eferience with {e} error")
        failure_counter.inc()
        raise e
        
    finally:
        elapsed_time = time.time() - start_time

        latency_histogram.observe(elapsed_time)

        results = {
            "prediction": prediction_value.tolist(),
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

