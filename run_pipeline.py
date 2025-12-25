"""
Main entry point for running pipelines
supports training, deployment, and inference modes
"""
import argparse
from pipelines.trainning_pipeline import training_pipeline
from pipelines.deployement_pipeline import deployment_pipeline


# data path - update this for your environment
DATA_PATH = "/Users/mac/Documents/zeamani/Learn/MLops/churny/extracted_data/customer_churn_dataset-testing-master.csv"


def run_training(model_name: str = "RandomForest", hyperparams: dict = None):
    """run training pipeline for experimentation"""
    print("=" * 50)
    print("Running TRAINING pipeline")
    print("=" * 50)
    
    metrics = training_pipeline(
        file_path=DATA_PATH,
        model_name=model_name,
        hyperparams=hyperparams,
        run_name=f"train_{model_name.lower()}"
    )
    
    print(f"\nTraining complete!")
    return metrics


def run_deployment(model_name: str = "GradientBoosting", hyperparams: dict = None, min_accuracy: float = 0.85):
    """run deployment pipeline for production"""
    print("=" * 50)
    print("Running DEPLOYMENT pipeline")
    print(f"Min accuracy threshold: {min_accuracy}")
    print("=" * 50)
    
    # production-ready hyperparams
    if hyperparams is None:
        hyperparams = {
            "gb_n_estimators": 100,
            "gb_learning_rate": 0.1,
            "gb_max_depth": 4
        }
    
    status = deployment_pipeline(
        file_path=DATA_PATH,
        model_name=model_name,
        hyperparams=hyperparams,
        min_accuracy=min_accuracy
    )
    
    print(f"\nDeployment status: {status}")
    return status


def run_inference(data_path: str = None, model_path: str = None):
    """run inference pipeline for predictions"""
    from pipelines.inference_pipeline import inference_pipeline
    
    print("=" * 50)
    print("Running INFERENCE pipeline")
    print("=" * 50)
    
    if data_path is None:
        data_path = DATA_PATH
    
    predictions = inference_pipeline(
        data_path=data_path,
        model_path=model_path
    )
    
    print(f"\nInference complete!")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipelines")
    parser.add_argument(
        "--mode", 
        choices=["train", "deploy", "inference"],
        default="deploy",
        help="Pipeline mode to run"
    )
    parser.add_argument(
        "--model",
        default="GradientBoosting",
        help="Model to use (RandomForest, LogisticRegression, GradientBoosting)"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.85,
        help="Minimum accuracy threshold for deployment (default: 0.85)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training(model_name=args.model)
    elif args.mode == "deploy":
        run_deployment(model_name=args.model, min_accuracy=args.min_accuracy)
    elif args.mode == "inference":
        run_inference()
