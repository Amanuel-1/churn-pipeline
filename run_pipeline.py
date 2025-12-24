from pipelines.trainning_pipeline import training_pipeline

if __name__ == "__main__":
    # Run the training pipeline
    file_path = "/Users/mac/Documents/zeamani/Learn/MLops/churny/extracted_data/customer_churn_dataset-testing-master.csv"
    
    print("Starting training pipeline...")
    
    # Run with RandomForest (default)
    metrics = training_pipeline(file_path=file_path, model_name="RandomForest")
    
    print("Training pipeline completed successfully!")
    print(f"""Final metrics:
         {metrics}
         """)
