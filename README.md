# Customer Churn Prediction MLOps Pipeline

A production-grade machine learning operations (MLOps) pipeline for predicting customer churn, featuring automated training, experiment tracking, model registry, and real-time inference through a web application.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-pipeline.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://dagshub.com/Amanuel-1/churn-pipeline.mlflow)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Pipeline Components](#pipeline-components)
- [Usage](#usage)
- [Model Registry](#model-registry)
- [Deployment](#deployment)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an end-to-end MLOps solution for customer churn prediction. It demonstrates industry best practices for:

- **Reproducible ML Pipelines**: Orchestrated workflows with ZenML
- **Experiment Tracking**: Comprehensive logging with MLflow on DagsHub
- **Model Versioning**: Centralized model registry for governance
- **Automated Deployment**: Quality-gated production deployments
- **Real-time Inference**: Interactive web application for predictions

### Key Features

- Automated data validation and preprocessing
- Multiple model training with hyperparameter optimization
- Quality gates ensuring only high-performing models reach production
- Real-time single and batch predictions
- Model performance monitoring and drift detection capabilities

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MLOps Pipeline Architecture                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   Data       │───▶│   Feature    │───▶│   Model      │───▶│  Model     │ │
│  │   Ingestion  │    │   Engineering│    │   Training   │    │  Evaluation│ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         ZenML Orchestration                              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    MLflow Experiment Tracking (DagsHub)                  ││
│  │         Parameters │ Metrics │ Artifacts │ Model Registry                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                      │                                       │
│                                      ▼                                       │
│                        ┌──────────────────────────┐                         │
│                        │   Quality Gate (≥85%)    │                         │
│                        └──────────────────────────┘                         │
│                                      │                                       │
│                          ┌───────────┴───────────┐                          │
│                          ▼                       ▼                          │
│                   ┌────────────┐          ┌────────────┐                    │
│                   │   Deploy   │          │   Reject   │                    │
│                   └────────────┘          └────────────┘                    │
│                          │                                                   │
│                          ▼                                                   │
│                   ┌────────────────────────────────┐                        │
│                   │   Streamlit Web Application    │                        │
│                   │   (Real-time Predictions)      │                        │
│                   └────────────────────────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **ML Framework** | scikit-learn | Model training and inference |
| **Pipeline Orchestration** | ZenML | ML workflow management |
| **Experiment Tracking** | MLflow | Metrics, parameters, and artifact logging |
| **Model Registry** | MLflow (DagsHub) | Model versioning and governance |
| **Data Processing** | Pandas, NumPy | Data manipulation and analysis |
| **Web Application** | Streamlit | Interactive prediction interface |
| **Cloud Platform** | DagsHub | MLflow hosting and collaboration |
| **Deployment** | Streamlit Cloud | Application hosting |
| **Version Control** | Git, DVC | Code and data versioning |

## Project Structure

```
churn-pipeline/
├── app.py                      # Streamlit web application
├── run_pipeline.py             # Main pipeline entry point
├── run_experiments.py          # Experiment runner for model comparison
├── requirements.txt            # Python dependencies
│
├── pipelines/
│   ├── trainning_pipeline.py   # Training pipeline definition
│   ├── deployement_pipeline.py # Deployment pipeline with quality gates
│   └── inference_pipeline.py   # Batch inference pipeline
│
├── steps/
│   ├── ingest_data.py          # Data ingestion step
│   ├── clean_data.py           # Data preprocessing step
│   ├── train_model.py          # Model training step
│   ├── evaluate_model.py       # Model evaluation step
│   ├── deployment_steps.py     # Deployment-specific steps
│   └── config.py               # Model configurations
│
├── src/
│   ├── ingest_util.py          # Data ingestion utilities
│   ├── clean_util.py           # Data cleaning utilities
│   ├── model_util.py           # Model training utilities
│   └── evaluation_util.py      # Evaluation metrics utilities
│
├── data/
│   └── customer_churn_dataset.zip
│
├── models/                     # Local model artifacts
├── mlruns/                     # Local MLflow tracking (development)
│
├── analysis/
│   └── churn_prediction.ipynb  # Exploratory data analysis
│
└── .streamlit/
    └── secrets.toml            # Streamlit secrets (not in git)
```

## Installation

### Prerequisites

- Python 3.10+
- pip or conda package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Amanuel-1/churn-pipeline.git
   cd churn-pipeline
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize ZenML**
   ```bash
   zenml init
   ```

5. **Configure DagsHub authentication** (for remote tracking)
   ```bash
   export DAGSHUB_USER_TOKEN="your-token"
   ```

## Pipeline Components

### Training Pipeline

The training pipeline handles model development and experimentation. It ingests raw data, validates and preprocesses it, trains the specified model, evaluates performance metrics, and logs everything to MLflow for tracking and comparison.

### Deployment Pipeline

The deployment pipeline includes quality gates to ensure only high-performing models reach production. Models must meet a minimum accuracy threshold (default 85%) before being registered in the model registry and deployed.

### Supported Models

| Model | Configuration Key | Default Hyperparameters |
|-------|-------------------|------------------------|
| Random Forest | `RandomForest` | n_estimators=100, max_depth=None |
| Logistic Regression | `LogisticRegression` | C=1.0, max_iter=100 |
| Gradient Boosting | `GradientBoosting` | n_estimators=100, learning_rate=0.1 |
| Support Vector Machine | `SVM` | C=1.0, kernel='rbf' |

## Usage

### Running the Training Pipeline

```bash
# Train with default settings (Gradient Boosting)
python run_pipeline.py --mode train

# Train with specific model
python run_pipeline.py --mode train --model RandomForest
```

### Running Experiments

Compare multiple models and hyperparameter configurations:

```bash
python run_experiments.py
```

This executes predefined experiments including:
- Random Forest (baseline, deep trees, shallow trees)
- Logistic Regression (baseline, high regularization)
- Gradient Boosting (baseline, slow learner, fast learner)

### Deploying a Model

```bash
# Deploy with default 85% accuracy threshold
python run_pipeline.py --mode deploy

# Deploy with custom threshold
python run_pipeline.py --mode deploy --min-accuracy 0.90
```

### Running Inference

```bash
python run_pipeline.py --mode inference
```

### Launching the Web Application

```bash
streamlit run app.py
```

## Model Registry

Models are registered and versioned in MLflow hosted on DagsHub:

- **Registry URL**: [https://dagshub.com/Amanuel-1/churn-pipeline.mlflow](https://dagshub.com/Amanuel-1/churn-pipeline.mlflow)
- **Model Name**: `churn_predictor_model`

### Model Lifecycle

1. **Training**: Models are trained and logged with metrics
2. **Evaluation**: Performance is assessed against quality thresholds
3. **Registration**: Passing models are registered in the model registry
4. **Deployment**: Registered models are deployed to production

## Deployment

### Streamlit Cloud Deployment

The application is deployed on Streamlit Cloud with the following configuration:

1. **Repository**: Connected to GitHub repository
2. **Main file**: `app.py`
3. **Requirements**: `requirements-streamlit.txt`

### Environment Variables

Configure the following secrets in Streamlit Cloud:

| Variable | Description |
|----------|-------------|
| `DAGSHUB_USER_TOKEN` | DagsHub authentication token |
| `DAGSHUB_USERNAME` | DagsHub username |

### Live Application

Access the deployed application: [https://churn-pipeline.streamlit.app](https://churn-pipeline.streamlit.app)

## API Reference

### Prediction Input Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| Gender | Categorical | Customer gender | Male, Female |
| Age | Integer | Customer age | 18-80 |
| Tenure | Integer | Months as customer | 1-60 |
| Usage Frequency | Integer | Monthly usage count | 1-30 |
| Support Calls | Integer | Support tickets raised | 0-10 |
| Payment Delay | Integer | Days of payment delay | 0-30 |
| Subscription Type | Categorical | Plan type | Basic, Standard, Premium |
| Contract Length | Categorical | Contract duration | Monthly, Quarterly, Annual |
| Total Spend | Float | Total amount spent ($) | 0-10000 |
| Last Interaction | Integer | Days since last interaction | 1-30 |

### Prediction Output

| Field | Type | Description |
|-------|------|-------------|
| Prediction | String | "Churn" or "No Churn" |
| Churn Probability | Float | Probability score (0.0 - 1.0) |
| Risk Factors | List | Identified risk factors for the customer |

## Metrics and Monitoring

### Model Performance Metrics

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Experiment Tracking

All experiments are tracked in MLflow with hyperparameters, performance metrics, model artifacts, and training metadata.

View experiments: [MLflow Dashboard](https://dagshub.com/Amanuel-1/churn-pipeline.mlflow)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new functionality
- Update documentation as needed
- Ensure all pipelines pass before submitting PR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ZenML](https://zenml.io/) for ML pipeline orchestration
- [MLflow](https://mlflow.org/) for experiment tracking
- [DagsHub](https://dagshub.com/) for MLflow hosting
- [Streamlit](https://streamlit.io/) for the web application framework

---

**Author**: Amanuel  
**Contact**: [GitHub](https://github.com/Amanuel-1)  
**Project Link**: [https://github.com/Amanuel-1/churn-pipeline](https://github.com/Amanuel-1/churn-pipeline)
