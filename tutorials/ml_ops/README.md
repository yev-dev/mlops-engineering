# MLOps: From Model Training to Production Deployment

> **A practical demonstration of MLOps, ML pipelines, model inferencing, deployment, monitoring, and drift detection using the mlops-engineering project architecture**

This tutorial covers the operational side of machine learning — how to take a trained model and deploy it to production with monitoring, alerting, and observability. It uses the [mlops-engineering](https://github.com/yev-dev/mlops-engineering) project as a reference architecture.

The companion notebook `ml_ops.ipynb` provides runnable Python implementations for all concepts discussed here.

---

## Table of Contents

1. [What is MLOps?](#1-what-is-mlops)
2. [Project Architecture Overview](#2-project-architecture-overview)
3. [Model Inferencing Frameworks](#3-model-inferencing-frameworks)
4. [Docker for Model Serialisation & Deployment](#4-docker-for-model-serialisation--deployment)
5. [Kubernetes for MLOps](#5-kubernetes-for-mlops)
6. [MLflow for Experiment Tracking & Model Registry](#6-mlflow-for-experiment-tracking--model-registry)
7. [Monitoring & Observability](#7-monitoring--observability)
8. [Drift Detection in Production](#8-drift-detection-in-production)
9. [CI/CD for ML Pipelines](#9-cicd-for-ml-pipelines)
10. [End-to-End Python Implementation](#10-end-to-end-python-implementation)

---

## 1. What is MLOps?

MLOps (Machine Learning Operations) is the practice of applying DevOps principles to machine learning workflows. It aims to:

- **Automate** the ML lifecycle from experimentation to production
- **Monitor** model performance and data quality continuously
- **Govern** model versions, experiments, and deployments
- **Collaborate** across data science, engineering, and operations teams

### The MLOps Maturity Model

```
Level 0: Manual                 → Jupyter notebooks, no versioning
Level 1: ML Pipeline Automation → CI/CD for training, model registry
Level 2: Full Automation        → Automated retraining, A/B testing, drift detection
Level 3: Continuous Operations  → Self-healing, auto-scaling, observability
```

The mlops-engineering project targets **Level 1–2**, with automated builds, containerised deployment, Prometheus monitoring, and Evidently drift detection.

### MLOps vs. Traditional DevOps

| Aspect | DevOps | MLOps |
|---|---|---|
| **Artifact** | Code (deterministic) | Code + Data + Model (non-deterministic) |
| **Testing** | Unit/integration tests | Data validation, model evaluation, bias checks |
| **Deployment** | Deploy code | Deploy model + inference service + feature pipeline |
| **Monitoring** | System metrics (CPU, latency) | System + Data drift + Concept drift + Model decay |
| **Retraining** | Not applicable | Automated retraining triggers |
| **Versioning** | Code versioned | Data + Model + Code all versioned |

---

## 2. Project Architecture Overview

The mlops-engineering project implements a complete MLOps stack with the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Client                            │
│                    POST /predict {"data": [85]}                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Inference Service                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  /predict endpoint                                         │  │
│  │  • Loads model.pkl from pickle                             │  │
│  │  • Runs sklearn LinearRegression.predict()                  │  │
│  │  • Returns {"prediction": 82.34, "timestamp": "..."}       │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Prometheus Metrics                                        │  │
│  │  • total_requests (Counter)                                │  │
│  │  • successful_predictions / failed_predictions (Counter)    │  │
│  │  • request_latency_seconds (Histogram)                      │  │
│  │  • model_prediction (Gauge)                                 │  │
│  │  • cpu_usage_percent / memory_usage_bytes (Gauge)           │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Drift Monitor (Evidently)                                  │  │
│  │  • PSI, KS-test on feature distributions                    │  │
│  │  • Regression performance tracking                          │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
┌─────────────────┐ ┌────────────┐ ┌──────────────┐
│   Prometheus    │ │  Grafana   │ │    MLflow    │
│  (metric store) │ │(dashboard) │ │ (registry)   │
│  port 9090      │ │ port 3000  │ │  (planned)   │
└─────────────────┘ └────────────┘ └──────────────┘
```

### Architecture Components

| Component | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI | High-performance async Python web framework for model serving |
| **Model Serialisation** | pickle | Serialise trained sklearn model to binary file |
| **Containerisation** | Docker | Package service + dependencies into portable image |
| **Orchestration** | docker-compose | Local multi-container deployment (app + monitoring) |
| **Metrics** | Prometheus | Time-series metric collection and alerting |
| **Visualisation** | Grafana | Dashboards for ML metrics (latency, drift, predictions) |
| **Drift Detection** | Evidently | Statistical drift monitoring (WIP) |
| **Model Registry** | MLflow | Experiment tracking and model versioning (planned) |
| **Object Storage** | MinIO | S3-compatible storage for model artifacts (planned) |

---

## 3. Model Inferencing Frameworks

Model inferencing is the process of using a trained model to make predictions on new data. Several frameworks can serve models in production:

### 3.1 Which Framework to Choose?

| Framework | Best For | Latency | Maturity | REST API |
|---|---|---|---|---|
| **FastAPI** | Python-native models, sklearn, PyTorch | Low (~5ms) | High | Built-in |
| **Flask** | Simple deployments, legacy projects | Medium | Very High | Manual |
| **TensorFlow Serving** | TF models, high throughput | Very Low | Very High | gRPC + REST |
| **TorchServe** | PyTorch models | Low | High | REST + gRPC |
| **ONNX Runtime** | Cross-framework, optimised inference | Very Low | High | Via FastAPI/Flask |
| **NVIDIA Triton** | GPU inference, multi-framework | Lowest | High | HTTP + gRPC |
| **BentoML** | Python model packaging + serving | Low | High | Built-in |

### 3.2 FastAPI in This Project

The project uses **FastAPI** for its async performance, automatic OpenAPI documentation, and first-class Pydantic validation.

```python
# From app/src/main.py — inference endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput, debug: bool = False):
    
    if loaded_model is None:
        raise RuntimeError("Model is not loaded.")
    
    request_counter.inc()
    
    start_time = time.time()
    try:
        prediction_value = loaded_model.predict([input_data.data])[0]
        prediction_gauge.set(prediction_value)
        prediction_histogram.observe(prediction_value)
        success_counter.inc()
    except Exception as e:
        failure_counter.inc()
        raise e
    
    elapsed_time = time.time() - start_time
    latency_histogram.observe(elapsed_time)
    
    return {
        "prediction": prediction_value.tolist(),
        "timestamp": datetime.datetime.now().isoformat()
    }
```

### 3.3 Model Serialisation Patterns

Models must be serialised (saved to disk) after training and deserialised (loaded) at inference time.

```python
# Serialisation with pickle (current project approach)
import pickle

with open("regression_model.pkl", "wb") as f:
    pickle.dump(trained_model, f)

# At inference time
with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)
```

**Better serialisation alternatives:**

```python
# joblib — better for large numpy arrays (sklearn recommended)
import joblib
joblib.dump(trained_model, "regression_model.joblib")
model = joblib.load("regression_model.joblib")

# ONNX — cross-platform, cross-language
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(trained_model, initial_types=initial_type)
with open("regression_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# MLflow — managed serialisation with registry
import mlflow.sklearn
mlflow.sklearn.log_model(trained_model, "model")
loaded = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

---

## 4. Docker for Model Serialisation & Deployment

Docker packages the model, its dependencies, and the inference service into a portable container image.

### 4.1 Why Docker for ML?

- **Reproducibility**: Same environment everywhere (dev, test, prod)
- **Portability**: Run on any Linux host, Kubernetes, or cloud
- **Isolation**: Model dependencies don't conflict with other services
- **Scalability**: Horizontal scaling with container orchestrators
- **CI/CD integration**: Build once, deploy anywhere

### 4.2 Dockerfile Breakdown

```dockerfile
# From app/Dockerfile
FROM python:3.11-slim          # Base image with Python

WORKDIR /app/                   # Working directory inside container

COPY . .                        # Copy source code + model + requirements

RUN pip install --no-cache-dir -r requirements.txt  # Install deps

EXPOSE 5000                    # Document the port

CMD ["uvicorn", "--host=0.0.0.0", "src.main:app", "--port=5000"]
```

### 4.3 Building and Running

```bash
# Build the image
docker build -t ml-ops:latest ./app

# Run the container
docker run --name=ml-ops --rm -p 5000:5000 ml-ops:latest

# Test inference
curl -X POST "http://127.0.0.1:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": [85]}'
# → {"prediction": 82.34, "timestamp": "2025-07-13T21:00:00"}
```

### 4.4 Multi-Stage Builds for Model Serving

For production, multi-stage builds reduce image size by separating build and runtime:

```dockerfile
# Stage 1: Build
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t /deps

# Stage 2: Runtime (tiny image)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
COPY data/regression_model.pkl ./data/
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]
```

### 4.5 Docker Compose for Full Stack

```yaml
# docker-compose.yaml
version: "3.8"
services:
  ml_app:
    build: ./app
    ports: ["5000:5000"]
    
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana-dashboard:
    image: grafana/grafana
    ports: ["3000:3000"]
    depends_on: [prometheus]
    env_file: [./grafana/config.monitoring]
```

---

## 5. Kubernetes for MLOps

Kubernetes (K8s) orchestrates containerised applications across a cluster of machines. It is the industry standard for production ML deployments.

### 5.1 Why Kubernetes for ML?

| Capability | Benefit for MLOps |
|---|---|
| **Auto-scaling** | Scale inference replicas based on request volume |
| **Self-healing** | Restart failed pods, replace unhealthy nodes |
| **Rolling updates** | Zero-downtime model updates (canary, blue-green) |
| **Resource management** | CPU/memory limits per model; GPU scheduling |
| **Service discovery** | Internal DNS for microservice communication |
| **Batch processing** | Jobs/CronJobs for scheduled model retraining |

### 5.2 Kubernetes Architecture for ML

```
┌───────────────────────────────────────────────────────┐
│                Kubernetes Cluster                      │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Inference   │  │ Inference   │  │ Inference   │   │
│  │ Pod (v1.0)  │  │ Pod (v1.0)  │  │ Pod (v1.1)  │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │            │
│         └────────────────┼────────────────┘            │
│                          │                             │
│                   ┌──────┴──────┐                      │
│                   │  Service    │                      │
│                   │ (LoadBalancer)                     │
│                   └──────┬──────┘                      │
│                          │                             │
│                   ┌──────┴──────┐                      │
│                   │   Ingress   │                      │
│                   │  (API GW)   │                      │
│                   └─────────────┘                      │
│                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Prometheus  │  │   Grafana   │  │   MLflow    │   │
│  │   Operator  │  │             │  │  Tracking   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘   │
└───────────────────────────────────────────────────────┘
```

### 5.3 Kubernetes Manifests for Model Serving

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
  labels:
    app: ml-inference
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
      - name: inference
        image: ml-ops:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /
            port: 5000
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /docs
            port: 5000
          initialDelaySeconds: 15
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-inference-service
spec:
  type: ClusterIP
  selector:
    app: ml-inference
  ports:
  - port: 80
    targetPort: 5000
---
# horizontal-pod-autoscaler.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 5.4 GPU Inference on Kubernetes

```yaml
# gpu-inference.yaml (for deep learning models)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gpu-inference
  template:
    metadata:
      labels:
        app: gpu-inference
    spec:
      containers:
      - name: inference
        image: nvidia/tritonserver:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001  # gRPC
        resources:
          limits:
            nvidia.com/gpu: 1   # Request 1 GPU
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

### 5.5 Canary Deployments for Model Updates

```yaml
# canary-deployment.yaml — route 10% traffic to new model version
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-inference-canary
  annotations:
    nginx.ingress.kubernetes.io/canary: "true"
    nginx.ingress.kubernetes.io/canary-weight: "10"
spec:
  rules:
  - http:
      paths:
      - path: /predict
        pathType: Prefix
        backend:
          service:
            name: ml-inference-v2
            port:
              number: 80
```

---

## 6. MLflow for Experiment Tracking & Model Registry

[MLflow](https://mlflow.org/) is an open-source platform for the complete ML lifecycle. It is planned for integration in the mlops-engineering project.

### 6.1 Why MLflow?

| Capability | What it Solves |
|---|---|
| **Tracking** | Log parameters, metrics, and artifacts for every experiment |
| **Model Registry** | Version models, promote stages (Staging → Production) |
| **Model Serving** | Deploy models as REST APIs with one command |
| **Project Packaging** | Reproducible runs with Conda/Docker environments |
| **Model Flavours** | Built-in support for sklearn, PyTorch, TF, ONNX, etc. |

### 6.2 MLflow Tracking

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("stock-price-prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("features", ["WTI_Oil"])
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    # Tag the run
    mlflow.set_tag("version", "v1.0.0")
    mlflow.set_tag("dataset", "sp500-2015-2025")

print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### 6.3 MLflow Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://mlflow-server:5000")

# Register model
result = mlflow.register_model(
    "runs:/<run_id>/model",
    "stock-price-predictor"
)

# Transition to staging
client.transition_model_version_stage(
    name="stock-price-predictor",
    version=1,
    stage="Staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="stock-price-predictor",
    version=1,
    stage="Production"
)

# Load from registry in production
import mlflow.sklearn
model = mlflow.sklearn.load_model(
    "models:/stock-price-predictor/Production"
)
```

### 6.4 MLflow + Docker Integration

```dockerfile
# Dockerfile with MLflow model server
FROM python:3.11-slim

# Install MLflow
RUN pip install mlflow scikit-learn pandas numpy

# Copy the serialised model
COPY regression_model.pkl /app/model/
COPY MLmodel /app/model/

# Serve via MLflow built-in server
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "5000", "--no-conda"]
```

### 6.5 MLflow + Kubernetes

```yaml
# mlflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow-server
  template:
    metadata:
      labels:
        app: mlflow-server
    spec:
      containers:
      - name: mlflow
        image: mlflow:latest
        args: ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000",
               "--backend-store-uri", "postgresql://user:pass@postgres/mlflow",
               "--default-artifact-root", "s3://mlflow-artifacts/"]
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_S3_ENDPOINT_URL
          value: "http://minio:9000"
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: minio-secret
              key: secret-key
```

---

## 7. Monitoring & Observability

Production ML systems require both **infrastructure monitoring** (CPU, memory, latency) and **ML-specific monitoring** (prediction distribution, data drift, model decay).

### 7.1 Prometheus Metrics

The project exposes metrics via `prometheus_client` on the `/metrics` endpoint:

```python
# From app/src/main.py — Prometheus metric definitions
from prometheus_client import Counter, Gauge, Histogram

# Gauges (point-in-time values)
cpu_usage_gauge = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_usage_gauge = Gauge("memory_usage_bytes", "Memory usage in bytes")
prediction_gauge = Gauge("model_prediction", "Model prediction value")
model_latency = Gauge("model_latency_seconds", "Time taken for model prediction")

# Counters (cumulative)
request_counter = Counter("total_requests", "Total number of prediction requests")
success_counter = Counter("successful_predictions", "Successful predictions")
failure_counter = Counter("failed_predictions", "Failed predictions")

# Histograms (distribution tracking)
latency_histogram = Histogram(
    "request_latency_seconds", "Request latency in seconds",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10)
)
prediction_histogram = Histogram(
    "prediction_values", "Distribution of prediction values"
)
```

### 7.2 Prometheus Configuration

```yaml
# prometheus/prometheus.yml
scrape_configs:
  - job_name: 'ml-app'
    scrape_interval: 5s
    static_configs:
      - targets: ['ml_app:5000']
```

### 7.3 Grafana Dashboard

The Grafana dashboard (pre-configured at `grafana/provisioning/dashboards/ml_metrics_dashboard.json`) visualises:

- **Request Rate**: Total requests per second
- **Latency**: P50, P95, P99 latency from the histogram
- **Prediction Values**: Distribution of model outputs over time
- **Error Rate**: Failed predictions / total requests
- **CPU & Memory**: Resource usage of the inference service
- **Drift Share**: Evidently data drift percentage (WIP)

```
╔═══════════════════════════════════════════════════════════════╗
║               ML Metrics Dashboard (Grafana)                  ║
╠═══════════════════════════════════════════════════════════════╣
║  ┌────────────────────┐  ┌────────────────────┐              ║
║  │ Request Rate: 12/s │  │ Error Rate: 0.5%  │              ║
║  └────────────────────┘  └────────────────────┘              ║
║  ┌──────────────────────────────────────────────────────────┐║
║  │  Request Latency (P95: 45ms, P99: 120ms)                 │║
║  │  ╱‾‾‾╲                                                     │║
║  │ ╱     ╲                                                    │║
║  └──────────────────────────────────────────────────────────┘║
║  ┌──────────────────────────────────────────────────────────┐║
║  │  Prediction Values Distribution                           │║
║  │  ████████████████░░░░░░░░░░░░░░░░░░░░                    │║
║  └──────────────────────────────────────────────────────────┘║
║  ┌────────────────────┐  ┌────────────────────┐              ║
║  │ Drift Share: 12%  │  │ Model Version: v2  │              ║
║  └────────────────────┘  └────────────────────┘              ║
╚═══════════════════════════════════════════════════════════════╝
```

![ML Metrics Dashboard](../README.md#grafana-ml-metrics-dashboard)

### 7.4 Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: ml_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(failed_predictions_total[5m]) / rate(total_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5% for 2 minutes"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(request_latency_seconds_bucket[5m])) > 1.0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeded 1 second"

      - alert: DataDriftDetected
        expr: model_drift_share > 0.25
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data drift share > 25% — consider retraining"
```

---

## 8. Drift Detection in Production

Drift detection is integrated into the model serving pipeline using **Evidently AI** and **SHAP**.

### 8.1 Drift Detection Pipeline

```
Training Data ──► Reference Distribution ──┐
                                           ├──► Evidently Report ──► Drift Metrics ──► Prometheus
Production Data ──► Current Distribution ──┘
                                              ──► SHAP Values ──► Feature Importance Stability
```

### 8.2 Evidently Integration

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset

def check_drift(reference_data, current_data):
    """Check data drift between training (reference) and production (current)."""
    
    # Data drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping={"numerical_features": ["WTI_Oil"]}
    )
    
    drift_metrics = drift_report.as_dict()
    drift_share = drift_metrics["metrics"][0]["result"]["drift_share"]
    
    # Regression performance report (if targets are available)
    perf_report = Report(metrics=[RegressionPreset()])
    perf_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping={
            "target": "XOM_actual",
            "prediction": "XOM_predicted",
            "numerical_features": ["WTI_Oil"]
        }
    )
    
    return {
        "drift_share": drift_share,
        "alert": drift_share > 0.25,
        "report": drift_report
    }
```

### 8.3 Drift Monitoring Class

```python
class ProductionDriftMonitor:
    """Monitor drift in production inference data."""
    
    def __init__(self, reference_data_path: str):
        self.reference_data = pd.read_csv(reference_data_path)
        self.drift_history = []
    
    def check_inference_drift(self, input_features: dict) -> dict:
        """Check drift for each prediction request (accumulates in buffer)."""
        self.buffer.append(input_features)
        
        if len(self.buffer) >= self.batch_size:
            batch_df = pd.DataFrame(self.buffer)
            
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(
                reference_data=self.reference_data,
                current_data=batch_df
            )
            
            drift_share = drift_report.as_dict()["metrics"][0]["result"]["drift_share"]
            self.drift_history.append({"drift_share": drift_share, "timestamp": datetime.now()})
            
            # Expose to Prometheus
            drift_share_gauge.set(drift_share)
            
            if drift_share > 0.25:
                logging.warning(f"Drift detected: {drift_share:.2%}")
                # Trigger retraining pipeline via webhook
                requests.post("http://retrainer:8000/retrain", json={"drift_share": drift_share})
            
            self.buffer = []
        
        return {"drift_share": drift_share if hasattr(self, 'buffer') else None}
```

### 8.4 Prometheus Drift Metrics

```python
# Expose drift metrics for Grafana alerting
from prometheus_client import Gauge

drift_share_gauge = Gauge("model_drift_share", "Data drift share (Evidently)")
model_rmse_gauge = Gauge("model_rmse", "Current RMSE on production data")
model_mae_gauge = Gauge("model_mae", "Current MAE on production data")
```

---

## 9. CI/CD for ML Pipelines

The project uses GitHub Actions for continuous integration and deployment.

### 9.1 CI/CD Pipeline Flow

```
┌─────────┐   ┌──────────┐   ┌────────┐   ┌─────────┐   ┌─────────┐
│  Code   │   │   Run    │   │ Build  │   │  Push   │   │ Deploy  │
│  Push   │──►│  Tests   │──►│ Docker │──►│ Registry│──►│  to K8s │
└─────────┘   └──────────┘   │ Image  │   └─────────┘   └─────────┘
                             └────────┘
```

### 9.2 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: ML Service CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install -r app/requirements.txt
    
    - name: Run tests
      run: pytest app/tests/
  
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t ml-ops:${{ github.sha }} ./app
    
    - name: Push to registry
      run: |
        docker tag ml-ops:${{ github.sha }} registry.example.com/ml-ops:latest
        docker push registry.example.com/ml-ops:latest
  
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/ml-inference \
          inference=registry.example.com/ml-ops:${{ github.sha }}
```

### 9.3 Model Validation Gate

```python
# tests/test_model_validation.py — CI gate before deployment
def test_model_performance():
    """Ensure model meets minimum performance threshold before deployment."""
    model = load_model()
    X_test, y_test = load_test_data()
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Model must pass these gates
    assert r2 > 0.7, f"R² score {r2:.4f} below minimum threshold 0.7"
    assert rmse < 15.0, f"RMSE {rmse:.4f} above maximum threshold 15.0"

def test_model_drift_gate():
    """Verify no significant drift between training and validation data."""
    psi = calculate_psi(train_features, val_features)
    assert psi < 0.25, f"PSI {psi:.4f} exceeds drift threshold 0.25"
```

---

## 10. End-to-End Python Implementation

The companion notebook `ml_ops.ipynb` contains runnable Python code that implements the entire MLOps pipeline described above. Here is a preview of what it covers:

### 10.1 Notebook Structure

| Section | Topic | Code Cells |
|---|---|---|
| 1 | FastAPI inference service (simulated) | 2 |
| 2 | Model serialisation (pickle, joblib, ONNX, MLflow) | 4 |
| 3 | Docker integration simulation | 1 |
| 4 | Kubernetes manifest generation | 2 |
| 5 | MLflow tracking & registry | 3 |
| 6 | Prometheus metric simulation | 3 |
| 7 | Drift detection with Evidently | 3 |
| 8 | Full CI/CD pipeline simulation | 2 |

### 10.2 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook ml_ops.ipynb

# Or for the full docker-compose stack (from project root)
docker compose up -d --build
```

### 10.3 Key Implementation: FastAPI Mock Server

```python
# From ml_ops.ipynb
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import time

app = FastAPI()

class PredictionInput(BaseModel):
    data: list

class PredictionOutput(BaseModel):
    prediction: float
    timestamp: str

# Load model (simulated)
with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    start = time.time()
    
    pred = model.predict([input_data.data])[0]
    
    return PredictionOutput(
        prediction=pred,
        timestamp=datetime.utcnow().isoformat()
    )
```

### 10.4 Key Implementation: Drift Detection System

```python
# From ml_ops.ipynb — full production drift monitor
class ProductionDriftMonitor:
    """End-to-end drift monitoring for production ML."""
    
    def __init__(self, model, reference_data, feature_names):
        self.model = model
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.buffer = []
        self.batch_size = 100
        self.drift_history = []
    
    def predict_and_monitor(self, features):
        """Make prediction and check drift."""
        # 1. Run inference
        prediction = self.model.predict([features])[0]
        
        # 2. Buffer for batch drift check
        self.buffer.append(features)
        
        # 3. Periodic drift check
        if len(self.buffer) >= self.batch_size:
            self._run_drift_check()
        
        return prediction
    
    def _run_drift_check(self):
        """Run Evidently drift check on buffered data."""
        current_df = pd.DataFrame(self.buffer, columns=self.feature_names)
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping={"numerical_features": self.feature_names}
        )
        
        result = report.as_dict()
        drift_share = result["metrics"][0]["result"]["drift_share"]
        
        self.drift_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "drift_share": drift_share,
            "alert": drift_share > 0.25
        })
        
        # Reset buffer
        self.buffer = []
        
        return drift_share
```

---

## Summary

The mlops-engineering project demonstrates the complete MLOps workflow:

| Stage | Implementation | Tool |
|---|---|---|
| **Model Training** | Linear regression on stock/oil data | scikit-learn |
| **Model Serialisation** | pickle binary format | pickle / joblib |
| **Inference API** | REST endpoint `/predict` | FastAPI |
| **Containerisation** | Docker image with dependencies | Docker |
| **Orchestration** | Multi-service deployment | docker-compose → Kubernetes |
| **Experiment Tracking** | Log params, metrics, artifacts | MLflow (planned) |
| **Model Registry** | Version control + stage promotion | MLflow (planned) |
| **Infra Monitoring** | CPU, memory, latency, error rate | Prometheus + Grafana |
| **ML Monitoring** | Feature drift, prediction drift | Evidently |
| **Explainability** | Feature importance, SHAP values | SHAP |
| **CI/CD** | Automated build, test, deploy | GitHub Actions |
| **Auto-scaling** | Horizontal scaling based on load | Kubernetes HPA |

### Why This Architecture Works

- **Separation of concerns**: Model training ↔ Serving ↔ Monitoring are independent
- **Observability by design**: Every prediction is instrumented with metrics
- **Drift-aware**: Continuous monitoring prevents silent model degradation
- **Production-ready**: Containerised, scalable, self-healing

### Next Steps for Production

1. Replace pickle with MLflow model registry for versioning
2. Add A/B testing infrastructure (canary deployments on K8s)
3. Implement automated retraining triggered by drift alerts
4. Add feature store (Feast/Tecton) for consistent feature computation
5. Deploy to Kubernetes with auto-scaling and GPU support

---

## References

- [mlops-engineering GitHub Repository](https://github.com/yev-dev/mlops-engineering)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Documentation](https://docs.docker.com/)