import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Model Prediction App", layout="wide")

# Title
st.title("🤖 Machine Learning Model Predictions")
st.write("Compare predictions from Model A and Model B")

# Load models
@st.cache_resource
def load_models():
    try:
        with open('../../Models/model_a.pkl', 'rb') as f:
            model_a = pickle.load(f)
        with open('../../Models/model_b.pkl', 'rb') as f:
            model_b = pickle.load(f)
        return model_a, model_b
    
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'model_a.pkl' and 'model_b.pkl' are in the same directory.")
        return None, None

model_a, model_b = load_models()

# Sidebar for model selection
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox(
    "Choose Model",
    ["Model A", "Model B", "Both Models"]
)

# Main content
if model_a is not None and model_b is not None:
    
    # Input method selection
    input_method = st.radio("Input Method:", ["Manual Input", "Upload CSV"])
    
    if input_method == "Manual Input":
        st.subheader("Enter Feature Values")
        
        # You'll need to adjust these based on your actual features
        # Example with 4 features
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            feature1 = st.number_input("Feature 1", value=0.0)
        with col2:
            feature2 = st.number_input("Feature 2", value=0.0)
        with col3:
            feature3 = st.number_input("Feature 3", value=0.0)
        with col4:
            feature4 = st.number_input("Feature 4", value=0.0)
        
        if st.button("Predict"):
            # Create input array
            input_data = np.array([[feature1, feature2, feature3, feature4]])
            
            # Make predictions
            st.subheader("Predictions")
            
            if selected_model == "Model A" or selected_model == "Both Models":
                pred_a = model_a.predict(input_data)
                st.success(f"**Model A Prediction:** {pred_a[0]}")
                
                # Show probability if classification
                if hasattr(model_a, 'predict_proba'):
                    proba_a = model_a.predict_proba(input_data)
                    st.write(f"Confidence: {max(proba_a[0]) * 100:.2f}%")
            
            if selected_model == "Model B" or selected_model == "Both Models":
                pred_b = model_b.predict(input_data)
                st.info(f"**Model B Prediction:** {pred_b[0]}")
                
                # Show probability if classification
                if hasattr(model_b, 'predict_proba'):
                    proba_b = model_b.predict_proba(input_data)
                    st.write(f"Confidence: {max(proba_b[0]) * 100:.2f}%")
    
    else:  # CSV Upload
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Generate Predictions"):
                # Make predictions
                if selected_model == "Model A" or selected_model == "Both Models":
                    df['Prediction_A'] = model_a.predict(df)
                
                if selected_model == "Model B" or selected_model == "Both Models":
                    df['Prediction_B'] = model_b.predict(df)
                
                st.subheader("Results")
                st.dataframe(df)
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

else:
    st.warning("Please ensure model files are loaded correctly.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Upload your trained models as 'model_a.pkl' and 'model_b.pkl'")