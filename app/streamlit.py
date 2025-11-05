import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Review Rating System",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern CSS with gradient backgrounds and cards
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        color: white;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* Input card */
    .input-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }
    
    /* Results cards */
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
        border: 2px solid transparent;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .model-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .confidence-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Agreement badge */
    .agreement-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        text-align: center;
        font-weight: 600;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    
    .difference-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        text-align: center;
        font-weight: 600;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load models (cached)
@st.cache_resource
def load_models():
    models = {}
    try:
        models['Model A'] = {
            'model': joblib.load('model_a.pkl'),
            'vectorizer': joblib.load('Model_A_vectorizer.pkl'),
            'loaded': True
        }
    except:
        models['Model A'] = {'loaded': False}
    
    try:
        models['Model B'] = {
            'model': joblib.load('model_b.pkl'),
            'vectorizer': joblib.load('Model_B_vectorizer.pkl'),
            'loaded': True
        }
    except:
        models['Model B'] = {'loaded': False}
    
    return models

# Prediction function
def predict_sentiment(text, model, vectorizer):
    if not text or len(text.strip()) == 0:
        return None, None
    
    try:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized)[0]
            confidence = max(probabilities) * 100
        
        return prediction, confidence
    except:
        return None, None

# Star rating display
def get_star_display(rating):
    full_stars = "⭐" * int(rating)
    return full_stars

# Main app
def main():
    # Header
    st.markdown("""
        <div class='header-container'>
            <div class='main-title'>Review Rating System</div>
            
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    loaded_models = [name for name, info in models.items() if info.get('loaded')]
    
    if not loaded_models:
        st.error("❌ No models found. Please check your model files.")
        st.stop()
    
   
    user_input = st.text_area(
        "📝 What's your review?",
        height=150,
        placeholder="Share your thoughts about the product... The more detailed, the better!",
        label_visibility="visible"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("✨ Analyze Sentiment", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis section
    if analyze_button:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review to analyze.")
        else:
            with st.spinner("🔮 Analyzing your review..."):
                # Get predictions
                results = {}
                for model_name in loaded_models:
                    model_info = models[model_name]
                    pred, conf = predict_sentiment(
                        user_input,
                        model_info['model'],
                        model_info['vectorizer']
                    )
                    results[model_name] = {'prediction': pred, 'confidence': conf}
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display results
                cols = st.columns(len(loaded_models))
                predictions = []
                
                for idx, model_name in enumerate(loaded_models):
                    with cols[idx]:
                        result = results[model_name]
                        
                        if result['prediction']:
                            pred = result['prediction']
                            predictions.append(pred)
                            
                            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                            st.markdown(f"<div class='model-name'>{model_name}</div>", unsafe_allow_html=True)
                            
                            st.markdown(f"""
                                <div style='text-align: center;'>
                                    <div class='score-badge'>{pred} / 5</div>
                                    <div style='font-size: 2rem; margin: 1rem 0;'>{get_star_display(pred)}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if result['confidence']:
                                conf = result['confidence']
                                st.markdown(f"""
                                    <div class='confidence-bar'>
                                        <div class='confidence-fill' style='width: {conf}%;'></div>
                                    </div>
                                    <div class='confidence-label'>Confidence: {conf:.1f}%</div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error("❌ Prediction failed")
                
                # Agreement check
                if len(predictions) > 1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if len(set(predictions)) == 1:
                        st.markdown("""
                            <div class='agreement-badge'>
                                ✅ Perfect Agreement! Both models predict the same rating.
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        diff = abs(predictions[0] - predictions[1])
                        st.markdown(f"""
                            <div class='difference-badge'>
                                📊 Models differ by {diff} star(s) • Consider both perspectives
                            </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()