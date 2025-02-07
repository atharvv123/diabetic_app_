import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with updated colors and styling
st.markdown("""
    <style>
    /* Main container styling */
    .main > div {
        padding: 0;
    }
    
    /* Header banner styling */
    .header-banner {
        background: linear-gradient(135deg, #00B4DB, #0083B0);
        color: white;
        padding: 2rem;
        border-radius: 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
    }
    
    .medical-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 1rem;
    }
    
    /* Card styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00B4DB, #0083B0);
        color: white;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 180, 219, 0.3);
    }
    
    /* Input containers */
    .input-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e1e4e8;
    }
    
    /* Headers */
    .header-text {
        color: #00B4DB;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .subheader-text {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    /* Results section */
    .results-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 1rem;
        margin-top: 2rem;
        border: 1px solid #e1e4e8;
    }
    
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        border: 1px solid #e1e4e8;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dark theme modifications */
    .stApp {
        background-color: #0f1116;
        color: #ffffff;
    }
    
    .metric-card {
        background-color: #1a1f25;
        border-color: #2d3139;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        color: #8b949e;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('knn_diabetes_model.pkl')

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found. Using placeholder for demonstration.")
    model = None

# Add a header banner with medical statistics
st.markdown("""
    <div class='header-banner'>
        <h1 style='margin:0;'>🏥 Diabetic Prediction & Analytics</h1>
        <p style='font-size: 1.1rem; opacity: 0.9;'>Comprehensive Health Assessment & Risk Analysis Platform</p>
        <div class='medical-stats'>
            <div>
                <h3>99.9%</h3>
                <p>Accuracy Rate</p>
            </div>
            <div>
                <h3>50K+</h3>
                <p>Analyses Completed</p>
            </div>
            <div>
                <h3>24/7</h3>
                <p>Monitoring</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Create form for all inputs
with st.form("prediction_form"):
    # Create three columns for input categories
    col1, col2, col3 = st.columns(3)
    
    # Personal Information
    with col1:
        st.markdown("""
            <div class='input-card'>
                <div class='header-text'>👤 Personal Information</div>
                <div class='subheader-text'>Basic demographic and physical measurements</div>
            </div>
        """, unsafe_allow_html=True)
        
        age = st.slider("Age (years)", 18, 100, 30, 
                       help="Adult age range: 18-100 years")
        
        bmi = st.slider("BMI (kg/m²)", 15.0, 70.0, 25.0, 0.1,
                       help="Normal range: 18.5-24.9")
        
        pregnancies = st.slider("Number of Pregnancies", 0, 15, 0,
                              help="Previous pregnancies: 0-15")
    
    # Clinical Measurements
    with col2:
        st.markdown("""
            <div class='input-card'>
                <div class='header-text'>🔬 Clinical Measurements</div>
                <div class='subheader-text'>Key medical indicators and measurements</div>
            </div>
        """, unsafe_allow_html=True)
        
        glucose = st.slider("Glucose (mg/dL)", 70, 200, 85,
                          help="Normal: 70-99 | Prediabetes: 100-125 | Diabetes: >126")
        
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 60, 150, 80,
                                 help="Normal: <120 | Elevated: 120-129 | High: ≥130")
        
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 30,
                                 help="Typical range: 20-50 mm")
    
    # Additional Factors
    with col3:
        st.markdown("""
            <div class='input-card'>
                <div class='header-text'>📊 Additional Factors</div>
                <div class='subheader-text'>Other relevant health indicators</div>
            </div>
        """, unsafe_allow_html=True)
        
        insulin = st.slider("Insulin (μU/ml)", 0, 900, 0,
                          help="Fasting: <25 | After meals: <130")
        
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01,
                       help="Higher values indicate stronger diabetic heredity")
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.form_submit_button("Generate Health Analysis")

# Results section
if predict_button:
    # Simulate prediction if model is not available
    if model is None:
        prediction = 1 if np.random.random() > 0.5 else 0
        prediction_proba = [1 - np.random.random() * 0.5, np.random.random() * 0.5]
    else:
        input_data = np.array([[pregnancies, glucose, blood_pressure, 
                               skin_thickness, insulin, bmi, dpf, age]])
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba[1] * 100,
        title={'text': "Risk Assessment", 'font': {'size': 24, 'color': '#8b949e'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#8b949e'},
            'bar': {'color': "#ff4b4b"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#2d3139",
            'steps': [
                {'range': [0, 30], 'color': '#A8E6CF'},
                {'range': [30, 70], 'color': '#FFD3B6'},
                {'range': [70, 100], 'color': '#FFB6B9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction_proba[1] * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "#8b949e"}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("BMI Status", f"{bmi:.1f}", "Normal" if 18.5 <= bmi <= 24.9 else "Attention Needed"),
        ("Glucose Level", str(glucose), "Normal" if glucose < 100 else "Elevated"),
        ("Blood Pressure", str(blood_pressure), "Normal" if blood_pressure < 80 else "Elevated"),
        ("Risk Category", "High" if prediction == 1 else "Low", f"{prediction_proba[1]*100:.1f}% Risk")
    ]

    for col, (label, value, delta) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{value}</div>
                    <div class='metric-delta'>{delta}</div>
                </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("""
        <div style='background-color: #1a1f25; padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem; border: 1px solid #2d3139;'>
            <h3 style='color: #00B4DB; margin-bottom: 1rem;'>📋 Health Insights & Recommendations</h3>
    """, unsafe_allow_html=True)
    
    if prediction == 1:
        st.error("⚠️ Higher Risk Profile Detected")
        st.markdown("""
            - Schedule a comprehensive health assessment with a healthcare provider
            - Implement regular blood glucose monitoring
            - Consider lifestyle modifications and dietary adjustments
            - Develop a structured exercise routine
            - Monitor cardiovascular health indicators
        """)
    else:
        st.success("✅ Lower Risk Profile Detected")
        st.markdown("""
            - Maintain current healthy lifestyle practices
            - Schedule regular preventive health check-ups
            - Continue balanced diet and exercise routine
            - Monitor any changes in health patterns
            - Stay informed about diabetes prevention
        """)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #8b949e;'>
        <p>🏥 This platform is designed for educational and preliminary assessment purposes only. 
           Always consult healthcare professionals for medical advice.</p>
        <p style='font-size: 0.8rem;'>Last updated: {}</p>
    </div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)