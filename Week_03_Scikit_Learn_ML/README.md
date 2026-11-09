# Week 3: Customer Churn Prediction Engine (Scikit-Learn, FastAPI & Docker)

An end-to-end machine learning project focusing on customer churn risk evaluation, Scikit-Learn pipelines, production-grade API deployment using FastAPI, and containerization with Docker.

## Live Demo
* **Streamlit Community Cloud:** [Customer Churn Prediction Engine](https://customer-churn-web-app-debn7cekcrda8mwprjtova.streamlit.app/app)

## Key Features & Capabilities
1. **Single Prediction Tab:** Interactive evaluation of individual customer churn risk based on tenure, monthly charges, and support calls with visual scoring.
2. **Batch Upload (CSV):** Bulk processing of customer records via CSV file upload to evaluate churn risk at scale.
3. **Model Insights:** Detailed overview of model architecture, feature importance charts, and input parameter breakdown.
4. **Production Backend:** FastAPI endpoint (`day_20_fastapi_app.py`) for real-time programmatic predictions.

## Project Structure
```text
Week_03_Scikit_Learn_ML/
│
├── saved_models/
│   └── churn_pipeline.joblib    # Serialized Scikit-Learn pipeline
├── day_15_supervised_ml.py      # Supervised workflow and linear models
├── day_16_classification.py     # Classification algorithms comparison
├── day_17_model_evaluation.py   # Evaluation metrics and 5-fold CV
├── day_18_hyperparameter_pipelines.py # GridSearchCV and ColumnTransformer
├── day_19_unsupervised_ml.py    # Unsupervised learning scripts
├── day_20_model_export.py       # Model training and persistence script
├── day_20_fastapi_app.py        # FastAPI application and Pydantic schemas
├── Dockerfile                   # Container configuration blueprint
└── requirements.txt             # Project Python dependencies
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