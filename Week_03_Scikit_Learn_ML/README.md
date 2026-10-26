# Week 3: Scikit-Learn Machine Learning & Production Deployment

This folder contains the practical implementation and code for Week 3 of the Machine Learning Engineering Sprint, focusing on machine learning pipelines, production-grade API deployment using FastAPI, and containerization with Docker.

## Overview

Week 3 bridges the gap between core model training and production-ready systems. The curriculum covers building automated Scikit-Learn workflows, persisting models with `joblib`, exposing real-time predictions through FastAPI endpoints, and containerizing the entire application using Docker for seamless deployment across any environment.

## Project Structure

```text
Week_03_Scikit_Learn_ML/
│
├── saved_models/
│   └── churn_pipeline.joblib          # Serialized Scikit-Learn pipeline
├── day_15_supervised_ml.py
├── day_16_classification.py
├── day_17_model_evaluation.py
├── day_18_hyperparameter_pipelines.py
├── day_19_unsupervised_ml.py
├── day_20_model_export.py             # Script to train and persist the model
├── day_20_fastapi_app.py              # FastAPI application and Pydantic schemas
├── Dockerfile                         # Container configuration blueprint
└── requirements.txt                   # Project Python dependencies
```

## How to Run Locally
``` Install the required dependencies:
pip install -r requirements.txt
```

## Run the FastAPI server:
```
uvicorn day_20_fastapi_app:app --reload
Open your browser and navigate to the interactive API docs:http://127.0.0.1:8000/docs
```

## Running with Docker
```
Build the Docker image:
docker build -t churn-api-image .
```

## Run the Docker container:
```
docker run -p 8000:8000 churn-api-image
```