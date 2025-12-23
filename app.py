import streamlit as st
import pandas as pd
import joblib
import numpy as np
from src.features import create_heart_disease_features
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Heart Disease Predictor", 
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E74C3C;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .low-risk {
        background-color: #D5F4E6;
        border: 2px solid #27AE60;
    }
    .high-risk {
        background-color: #FADBD8;
        border: 2px solid #E74C3C;
    }
    .metric-card {
        background-color: #F8F9F9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    """Load trained model, scaler, and feature list"""
    try:
        model = joblib.load('models/heart_disease_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        features = joblib.load('models/features_list.pkl')
        return model, scaler, features, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

model, scaler, expected_features, models_loaded = load_assets()

st.markdown('<p class="main-header"> Cardiovascular Risk Assessment</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Heart Disease Prediction System</p>', unsafe_allow_html=True)

with st.expander("About This Tool"):
    st.markdown("""
    ### How It Works
    This tool uses a **Logistic Regression model** trained on clinical cardiovascular data to assess heart disease risk.
    
    **Important Disclaimers:**
    - This is a **demonstration tool** for educational purposes
    - Not a substitute for professional medical advice
    - Results should be discussed with a qualified healthcare provider
    - Based on historical data and may not reflect individual circumstances
    """)

st.sidebar.header(" Patient Information")
st.sidebar.markdown("---")

def user_input_features():
    """Collect patient data from sidebar inputs"""
    
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age (years)", 20, 90, 50, help="Patient's age in years")
    sex = st.sidebar.selectbox(
        "Sex", 
        options=[1, 0], 
        format_func=lambda x: "Male" if x == 1 else "Female",
        help="Biological sex"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Symptoms & Examination")
    
    cp = st.sidebar.selectbox(
        "Chest Pain Type", 
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Typical Angina",
            2: "2 - Atypical Angina", 
            3: "3 - Non-Anginal Pain",
            4: "4 - Asymptomatic"
        }[x],
        help="Type of chest pain experienced"
    )
    
    exang = st.sidebar.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Chest pain triggered by exercise"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Vital Signs")
    
    trestbps = st.sidebar.number_input(
        "Resting Blood Pressure (mm Hg)", 
        90, 200, 120,
        help="Blood pressure at rest (normal: 90-120)"
    )
    
    chol = st.sidebar.number_input(
        "Serum Cholesterol (mg/dl)", 
        100, 600, 200,
        help="Total cholesterol (normal: <200)"
    )
    
    fbs = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "Yes (>120)" if x == 1 else "No (≤120)",
        help="Elevated fasting blood sugar indicator"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cardiac Tests")
    
    thalach = st.sidebar.slider(
        "Max Heart Rate Achieved (bpm)", 
        60, 220, 150,
        help="Maximum heart rate during stress test"
    )
    
    oldpeak = st.sidebar.slider(
        "ST Depression (oldpeak)", 
        0.0, 6.0, 1.0, 0.1,
        help="ST depression induced by exercise vs. rest"
    )
    
    restecg = st.sidebar.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Normal",
            1: "1 - ST-T Wave Abnormality",
            2: "2 - Left Ventricular Hypertrophy"
        }[x],
        help="Resting electrocardiographic results"
    )
    
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - Upsloping",
            2: "2 - Flat",
            3: "3 - Downsloping"
        }[x],
        help="Slope pattern during peak exercise"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced Tests")
    
    ca = st.sidebar.selectbox(
        "Major Vessels Colored (fluoroscopy)",
        options=[0, 1, 2, 3],
        help="Number of major vessels colored by fluoroscopy (0-3)"
    )
    
    thal = st.sidebar.selectbox(
        "Thalassemia",
        options=[3, 6, 7],
        format_func=lambda x: {
            3: "3 - Normal",
            6: "6 - Fixed Defect",
            7: "7 - Reversible Defect"
        }[x],
        help="Blood disorder test result"
    )
    
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Risk Prediction")

if not models_loaded:
    st.error("Models not loaded. Please ensure model files are in the 'models/' directory.")
    st.stop()

if st.button("Analyze Risk", type="primary"):
    with st.spinner("Analyzing patient data..."):
        try:
            processed_df = create_heart_disease_features(input_df)
            
            label_encode_cols = ['sex', 'fbs', 'exang', 'slope']
            onehot_cols = ['cp', 'restecg', 'ca', 'thal']
            
            for col in onehot_cols:
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].astype(str)
            
            processed_df = pd.get_dummies(processed_df, columns=onehot_cols, prefix=onehot_cols)
            
            for col in expected_features:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            processed_df = processed_df[expected_features]
            
            scaled_data = scaler.transform(processed_df)
            
            prediction = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)
            
            if prediction[0] == 0:
                confidence = probabilities[0][0] * 100
                st.markdown(f"""
                <div class="risk-box low-risk">
                    <h2 style="color: #27AE60;">Low Risk (Healthy)</h2>
                    <h1 style="color: #27AE60;">{confidence:.1f}%</h1>
                    <p style="color: #27AE60;">Confidence in healthy classification</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("The patient shows indicators consistent with low cardiovascular disease risk.")
                
            else:
                disease_prob = (1 - probabilities[0][0]) * 100
                severity = prediction[0]
                
                st.markdown(f"""
                <div class="risk-box high-risk">
                    <h2 style="color: #E74C3C;">Elevated Risk Detected</h2>
                    <h1 style="color: #E74C3C;">{disease_prob:.1f}%</h1>
                    <p style="color: #E74C3C;">Probability of heart disease</p>
                    <p style="color: #E74C3C; font-size: 1.2rem;">Severity Level: {severity}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning(f"The patient shows indicators consistent with heart disease (Severity Level {severity}).")
            
            st.subheader("Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Class': [f'Class {i}' for i in range(len(probabilities[0]))],
                'Probability': probabilities[0] * 100
            })
            
            fig = px.bar(
                prob_df, 
                x='Class', 
                y='Probability',
                color='Probability',
                color_continuous_scale='RdYlGn_r',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                xaxis_title="Disease Severity",
                yaxis_title="Probability (%)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if hasattr(model, 'coef_'):
                st.subheader("Top Risk Factors")
                
                if len(model.coef_.shape) > 1:
                    coef = np.mean(np.abs(model.coef_), axis=0)
                else:
                    coef = np.abs(model.coef_)
                
                feature_importance = pd.DataFrame({
                    'Feature': expected_features,
                    'Importance': coef * scaled_data[0]
                }).sort_values('Importance', ascending=False).head(10)
                
                fig2 = px.bar(
                    feature_importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                fig2.update_layout(
                    xaxis_title="Contribution to Prediction",
                    yaxis_title="",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7F8C8D; padding: 20px;">
    <p><strong>Heart Disease Risk Assessment Tool</strong></p>
    <p>Developed with ❤️ using Streamlit and scikit-learn</p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Resources
- [Project GitHub](https://github.com/martasolerebri/heart-disease-prediction.git)
""")
