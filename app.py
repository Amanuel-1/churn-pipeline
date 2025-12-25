"""
Streamlit app for churn prediction
uses the deployed model from MLflow registry (Dagshub)
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

# Set MLflow credentials BEFORE importing mlflow
DAGSHUB_USERNAME = "Amanuel-1"
DAGSHUB_REPO = "churn-pipeline"

# Get token from secrets or environment
token = None
if hasattr(st, 'secrets') and 'DAGSHUB_USER_TOKEN' in st.secrets:
    token = st.secrets['DAGSHUB_USER_TOKEN']
elif 'DAGSHUB_USER_TOKEN' in os.environ:
    token = os.environ['DAGSHUB_USER_TOKEN']

if token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token

import mlflow
import mlflow.sklearn
import joblib
import json
from glob import glob

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_REGISTRY_NAME = "churn_predictor"


def load_model_from_mlflow():
    """load model from mlflow registry"""
    
    # Known good run ID from Dagshub
    KNOWN_RUN_ID = "c9450f585d904694b7ac57bcf0e080b7"
    
    # Debug: show auth status
    st.sidebar.write("**Debug Info:**")
    st.sidebar.write(f"Auth configured: {'Yes' if token else 'No'}")
    
    client = mlflow.tracking.MlflowClient()
    
    # Method 1: Try model registry
    try:
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/1"
        model = mlflow.sklearn.load_model(model_uri)
        
        try:
            model_version = client.get_model_version(MODEL_REGISTRY_NAME, "1")
            run = client.get_run(model_version.run_id)
            metadata = {
                "model_params": run.data.params,
                "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                "run_id": model_version.run_id,
                "version": "1",
                "source": "mlflow_registry"
            }
        except:
            metadata = {"source": "mlflow_registry", "version": "1"}
        
        return model, metadata
    except Exception as e:
        st.sidebar.write(f"Registry: {str(e)[:50]}...")
    
    # Method 2: Load directly from known run ID
    try:
        model_uri = f"runs:/{KNOWN_RUN_ID}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        try:
            run = client.get_run(KNOWN_RUN_ID)
            metadata = {
                "model_params": run.data.params,
                "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                "run_id": KNOWN_RUN_ID,
                "source": "mlflow_run"
            }
        except:
            metadata = {"source": "mlflow_run", "run_id": KNOWN_RUN_ID}
        
        st.sidebar.write("✓ Loaded from run")
        return model, metadata
    except Exception as e:
        st.sidebar.write(f"Run load: {str(e)[:50]}...")
    
    # Method 3: Search experiments
    try:
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=3)
            for run in runs:
                try:
                    model_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.sklearn.load_model(model_uri)
                    metadata = {
                        "model_params": run.data.params,
                        "metrics": {k: float(v) for k, v in run.data.metrics.items()},
                        "run_id": run.info.run_id,
                        "source": "mlflow_run"
                    }
                    return model, metadata
                except:
                    continue
    except Exception as e:
        st.sidebar.write(f"Search: {str(e)[:50]}...")
    
    return None, None


def load_model_from_local():
    """fallback - load from local files"""
    model_dirs = glob("models/deployed_*")
    if not model_dirs:
        model_dirs = glob("models/production_*")
    
    if not model_dirs:
        return None, None
    
    latest_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(latest_dir, "model.joblib")
    metadata_path = os.path.join(latest_dir, "metadata.json")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            metadata["source"] = "local_file"
        return model, metadata
    
    return None, None


def load_model():
    """try mlflow first, fallback to local"""
    model, metadata = load_model_from_mlflow()
    if model is not None:
        return model, metadata
    
    # fallback
    return load_model_from_local()


def preprocess_input(data: dict) -> pd.DataFrame:
    """preprocess user input - must match training exactly"""
    
    # encode categorical values
    gender_encoded = 1 if data['Gender'] == 'Male' else 0
    subscription_encoded = {'Basic': 0, 'Standard': 1, 'Premium': 2}[data['Subscription Type']]
    contract_encoded = {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}[data['Contract Length']]
    
    # feature engineering
    tenure = data['Tenure'] if data['Tenure'] > 0 else 1
    spend_per_tenure = data['Total Spend'] / tenure
    support_call_rate = data['Support Calls'] / tenure
    usage_per_tenure = data['Usage Frequency'] / tenure
    
    # EXACT order from model.feature_names_in_:
    # ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay',
    #  'Subscription Type', 'Contract Length', 'Total Spend', 'Last Interaction',
    #  'Spend_per_Tenure', 'Support_Call_Rate', 'Usage_per_Tenure']
    df = pd.DataFrame([{
        'Age': data['Age'],
        'Gender': gender_encoded,
        'Tenure': data['Tenure'],
        'Usage Frequency': data['Usage Frequency'],
        'Support Calls': data['Support Calls'],
        'Payment Delay': data['Payment Delay'],
        'Subscription Type': subscription_encoded,
        'Contract Length': contract_encoded,
        'Total Spend': data['Total Spend'],
        'Last Interaction': data['Last Interaction'],
        'Spend_per_Tenure': spend_per_tenure,
        'Support_Call_Rate': support_call_rate,
        'Usage_per_Tenure': usage_per_tenure
    }])
    
    return df


def main():
    st.set_page_config(
        page_title="Churn Predictor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔮 Customer Churn Prediction")
    st.markdown("Predict whether a customer will churn based on their profile")
    
    # load model from mlflow
    model, metadata = load_model()
    
    if model is None:
        st.error("No deployed model found! Run the deployment pipeline first.")
        st.code("python run_pipeline.py --mode deploy")
        return
    
    # show model info in sidebar
    with st.sidebar:
        st.header("Model Info")
        if metadata:
            source = metadata.get('source', 'unknown')
            st.write(f"**Source:** {source}")
            
            if source == "mlflow_registry":
                st.write(f"**Version:** {metadata.get('version', 'N/A')}")
                st.write(f"**Run ID:** {metadata.get('run_id', 'N/A')[:8]}...")
            
            model_type = metadata.get('model_params', {}).get('model_type', 'Unknown')
            st.write(f"**Type:** {model_type}")
            
            metrics = metadata.get('metrics', {})
            st.write(f"**Accuracy:** {metrics.get('accuracy', 0):.2%}")
            st.write(f"**F1 Score:** {metrics.get('f1_score', 0):.2%}")
        else:
            st.write("Metadata not available")
    
    # input form
    st.header("Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 80, 35)
        tenure = st.slider("Tenure (months)", 1, 60, 12)
        usage_freq = st.slider("Usage Frequency", 1, 30, 15)
    
    with col2:
        support_calls = st.slider("Support Calls", 0, 10, 2)
        payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    
    with col3:
        contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend ($)", 0, 10000, 500)
        last_interaction = st.slider("Days Since Last Interaction", 1, 30, 10)
    
    # prepare input
    input_data = {
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Usage Frequency': usage_freq,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Subscription Type': subscription,
        'Contract Length': contract,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }
    
    st.markdown("---")
    
    # predict button
    if st.button("🎯 Predict Churn", type="primary"):
        with st.spinner("Making prediction..."):
            # preprocess
            features = preprocess_input(input_data)
            
            # predict
            prediction = model.predict(features)[0]
            
            # get probability if available
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
            
            # display result
            st.markdown("---")
            st.header("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ HIGH CHURN RISK")
                    st.markdown("This customer is **likely to churn**")
                else:
                    st.success("✅ LOW CHURN RISK")
                    st.markdown("This customer is **likely to stay**")
            
            with col2:
                if proba is not None:
                    churn_prob = proba[1] * 100
                    st.metric("Churn Probability", f"{churn_prob:.1f}%")
                    st.progress(proba[1])
            
            # risk factors
            st.markdown("---")
            st.subheader("Risk Analysis")
            
            risk_factors = []
            if support_calls >= 5:
                risk_factors.append("� Hoigh support calls")
            if payment_delay >= 15:
                risk_factors.append("� Signifigcant payment delays")
            if tenure <= 6:
                risk_factors.append("🟡 Short tenure")
            if contract == "Monthly":
                risk_factors.append("🟡 Monthly contract (less commitment)")
            if usage_freq <= 5:
                risk_factors.append("🟡 Low usage frequency")
            
            if risk_factors:
                st.write("**Identified risk factors:**")
                for factor in risk_factors:
                    st.write(f"  {factor}")
            else:
                st.write("✅ No major risk factors identified")
    
    # batch prediction
    st.markdown("---")
    st.header("📁 Batch Prediction")
    
    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing..."):
                    results = []
                    for _, row in batch_df.iterrows():
                        input_dict = {
                            'Gender': row.get('Gender', 'Male'),
                            'Age': row.get('Age', 30),
                            'Tenure': row.get('Tenure', 12),
                            'Usage Frequency': row.get('Usage Frequency', 15),
                            'Support Calls': row.get('Support Calls', 2),
                            'Payment Delay': row.get('Payment Delay', 5),
                            'Subscription Type': row.get('Subscription Type', 'Basic'),
                            'Contract Length': row.get('Contract Length', 'Monthly'),
                            'Total Spend': row.get('Total Spend', 500),
                            'Last Interaction': row.get('Last Interaction', 10)
                        }
                        features = preprocess_input(input_dict)
                        pred = model.predict(features)[0]
                        prob = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else None
                        results.append({
                            'Prediction': 'Churn' if pred == 1 else 'No Churn',
                            'Churn_Probability': f"{prob*100:.1f}%" if prob else "N/A"
                        })
                    
                    results_df = pd.DataFrame(results)
                    combined = pd.concat([batch_df.reset_index(drop=True), results_df], axis=1)
                    
                    st.write("Results:")
                    st.dataframe(combined)
                    
                    csv = combined.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
