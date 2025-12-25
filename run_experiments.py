"""
Run multiple training experiments with different models and hyperparameters
logs everything to mlflow for comparison
"""
from pipelines.trainning_pipeline import training_pipeline

# data path
FILE_PATH = "/Users/mac/Documents/zeamani/Learn/MLops/churny/extracted_data/customer_churn_dataset-testing-master.csv"


# experiments to run
EXPERIMENTS = [
    # Random Forest experiments
    {
        "model_name": "RandomForest",
        "run_name": "rf_baseline",
        "hyperparams": {
            "rf_n_estimators": 100,
            "rf_max_depth": None,
            "rf_min_samples_split": 2
        }
    },
    {
        "model_name": "RandomForest",
        "run_name": "rf_deep_trees",
        "hyperparams": {
            "rf_n_estimators": 200,
            "rf_max_depth": 20,
            "rf_min_samples_split": 5
        }
    },
    {
        "model_name": "RandomForest",
        "run_name": "rf_shallow_trees",
        "hyperparams": {
            "rf_n_estimators": 150,
            "rf_max_depth": 10,
            "rf_min_samples_split": 10
        }
    },
    
    # Logistic Regression experiments
    {
        "model_name": "LogisticRegression",
        "run_name": "lr_baseline",
        "hyperparams": {
            "lr_C": 1.0,
            "lr_max_iter": 100,
            "lr_solver": "lbfgs"
        }
    },
    {
        "model_name": "LogisticRegression",
        "run_name": "lr_high_regularization",
        "hyperparams": {
            "lr_C": 0.1,
            "lr_max_iter": 200,
            "lr_solver": "lbfgs"
        }
    },
    
    # skipping SVM - too slow for this dataset size
    
    # Gradient Boosting experiments
    {
        "model_name": "GradientBoosting",
        "run_name": "gb_baseline",
        "hyperparams": {
            "gb_n_estimators": 100,
            "gb_learning_rate": 0.1,
            "gb_max_depth": 3
        }
    },
    {
        "model_name": "GradientBoosting",
        "run_name": "gb_slow_learner",
        "hyperparams": {
            "gb_n_estimators": 200,
            "gb_learning_rate": 0.05,
            "gb_max_depth": 5
        }
    },
    {
        "model_name": "GradientBoosting",
        "run_name": "gb_fast_learner",
        "hyperparams": {
            "gb_n_estimators": 50,
            "gb_learning_rate": 0.2,
            "gb_max_depth": 4
        }
    },
]


def run_all_experiments():
    """run all experiments"""
    print("=" * 60)
    print("EXPERIMENT SUITE")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print("=" * 60)
    
    results = []
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp['run_name']}")
        print(f"  Model: {exp['model_name']}")
        
        try:
            metrics = training_pipeline(
                file_path=FILE_PATH,
                model_name=exp["model_name"],
                hyperparams=exp["hyperparams"],
                run_name=exp["run_name"]
            )
            
            results.append({
                "run": exp["run_name"],
                "model": exp["model_name"],
                "status": "ok"
            })
            print(f"  ✓ Done")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                "run": exp["run_name"],
                "status": "failed"
            })
    
    # summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["status"] == "ok")
    print(f"Completed: {passed}/{len(EXPERIMENTS)}")
    print("\nView results: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    run_all_experiments()
