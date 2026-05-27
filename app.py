import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import altair as alt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================================
# INITIAL SETUP & CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Mental Health Sentiment AI",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model_name = 'upgraded_RNN_model.h5'
    tok_name = 'tokenizer.pkl'
    enc_name = 'label_encoder.pkl'
    
    if os.path.exists(model_name):
        model_path = model_name
        tok_path = tok_name
        enc_path = enc_name
    else:
        model_path = os.path.join('models', model_name)
        tok_path = os.path.join('models', tok_name)
        enc_path = os.path.join('models', enc_name)
        
    if not os.path.exists(model_path):
        st.error(f"❌ Could not find model file: '{model_name}'. Please ensure it is uploaded to your project directory.")
        st.stop()
        
    model = load_model(model_path)
    with open(tok_path, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(enc_path, 'rb') as f:
        label_encoder = pickle.load(f)
        
    return model, tokenizer, label_encoder

try:
    model, tokenizer, label_encoder = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading deployment artifacts: {e}")
    st.stop()

# =====================================================================
# SECTION 1 — HEADER
# =====================================================================
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>AI-Based Mental Health Sentiment Monitoring System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #4B5563; font-weight: normal; margin-bottom: 30px;'>Emotion Detection using Simple Recurrent Neural Networks</h4>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================================
# SECTION 2 — ABOUT THE PROJECT
# =====================================================================
with st.expander("📖 About the Project & Technology Stack", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧠 Importance of Emotional AI & NLP")
        st.write(
            "Emotional AI bridges the gap between raw textual computation and human empathy. "
            "By deploying Natural Language Processing (NLP), digital health platforms can interpret the latent "
            "psychological markers embedded in human speech, scaling triage services and providing objective assessment tools."
        )
    with col2:
        st.markdown("### 🔄 Role of RNNs in Sequence Learning")
        st.write(
            "Unlike traditional algorithms that look at words out of order, Recurrent Neural Networks (RNNs) "
            "maintain a sequential memory via an internal hidden state vector. This sequential learning behavior "
            "allows the network to grasp context, grammatical shifting, and structural nuance over extended textual timelines."
        )

# =====================================================================
# SECTION 3 — USER TEXT INPUT AREA
# =====================================================================
st.markdown("### 📝 Enter Text Data for Triage")

# REMOVED: Sample dropdown options completely omitted. 
# Text field is blank and relies purely on user custom string arguments.
user_input = st.text_area(
    label="Input message conversation context:",
    value="",
    placeholder="Enter your thoughts or feelings here...",
    height=150
)

# =====================================================================
# SECTION 4 — PREDICTION BUTTON
# =====================================================================
submit_clicked = st.button("Analyze Emotion", type="primary", use_container_width=True)

# =====================================================================
# INFERENCE & RENDERING PIPELINE
# =====================================================================
if submit_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please input text first before clicking analyze.")
    else:
        # Preprocess input
        cleaned = user_input.lower()
        cleaned = re.sub(r'[^a-z\s]', '', cleaned)
        
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
        
        # Optimized Direct Call Pass (Prevents hanging/freezing on local machine)
        tensor_input = tf.convert_to_tensor(padded, dtype=tf.int32)
        probabilities = model(tensor_input, training=False).numpy()[0]
        
        predicted_idx = np.argmax(probabilities)
        emotion = label_encoder.classes_[predicted_idx]
        confidence = probabilities[predicted_idx] * 100
        
        # Format layout into columns
        out_col, viz_col = st.columns([1, 1])
        
        with out_col:
            # =====================================================================
            # SECTION 5 — PREDICTION OUTPUT
            # =====================================================================
            st.markdown("### 🎯 Diagnosis Output")
            
            st.info(f"**Emotion Detected:** {emotion}")
            st.metric(label="Confidence Level", value=f"{confidence:.1f}%")
            
            # Status badge determination
            if emotion in ["Depression", "Anxiety", "Panic"]:
                st.error("🚨 Emotional Status: Urgent Attention/Support Indicated")
            elif emotion in ["Stress", "Anger", "Sadness"]:
                st.warning("⚠️ Emotional Status: Moderate Distress Flagged")
            else:
                st.success("✨ Emotional Status: Positive / Balanced Emotional Well-being")
            
            # =====================================================================
            # SECTION 7 — EMOTIONAL GUIDANCE AREA
            # =====================================================================
            st.markdown("### 🌿 Emotional Wellness Guidance")
            
            guidance_data = {
                "Depression": {
                    "msg": "You do not have to carry this heavy weight entirely alone.",
                    "activity": "Reach out directly to a trusted friend, family member, or mental health advocate.",
                    "tips": ["Break larger goals down into tiny steps.", "Celebrate minor tasks like hydrating or resting."]
                },
                "Anxiety": {
                    "msg": "Your mind is moving rapidly right now. Let's work on grounding your attention.",
                    "activity": "Try the 5-4-3-2-1 sensory awareness method to anchor yourself.",
                    "tips": ["Focus on a slow 4-count box-breathing cycle.", "Limit caffeine and sensory overload."]
                },
                "Panic": {
                    "msg": "This intense somatic response is temporary. You are safe in this current immediate moment.",
                    "activity": "Place your hands flat on a cool surface or splash cold water on your face.",
                    "tips": ["Acknowledge the surge without fighting it.", "Remind yourself the feeling will pass shortly."]
                },
                "Stress": {
                    "msg": "Your cognitive load has outpaced your current resource availability.",
                    "activity": "Step away completely from your screen or assignments for a full 15-minute reset.",
                    "tips": ["List tasks by immediate tactical urgency.", "Practice saying no to extra obligations today."]
                },
                "Anger": {
                    "msg": "Anger is an indicator that a personal boundary may have been crossed.",
                    "activity": "Channel physical frustration cleanly through brisk walking or journaling thoughts.",
                    "tips": ["Take a brief physical pause before speaking.", "Reflect on the underlying vulnerability."]
                },
                "Sadness": {
                    "msg": "Allowing yourself to feel grief or sorrow is a natural step toward internal healing.",
                    "activity": "Listen to comforting media or allow yourself the tears needed to process.",
                    "tips": ["Do not force a forced smile if it feels dishonest.", "Treat yourself with gentle patience."]
                },
                "Happy": {
                    "msg": "It is wonderful to celebrate these clear, resilient moments of internal equilibrium!",
                    "activity": "Write down this specific memory to look back on when navigating future clouds.",
                    "tips": ["Share your uplifting energy with someone else.", "Continue nourishing your self-care habits."]
                }
            }
            
            active_guide = guidance_data.get(emotion, {"msg": "Take a short break and talk with someone you trust.", "activity": "Engage in mindfulness.", "tips": ["Prioritize sleep."]})
            
            st.markdown(f"*\"{active_guide['msg']}\"*")
            st.markdown(f"**💡 Suggested Activity:** {active_guide['activity']}")
            st.markdown("**📌 Daily Wellness Tips:**")
            for tip in active_guide['tips']:
                st.markdown(f"- {tip}")
                
        with viz_col:
            # =====================================================================
            # SECTION 6 — VISUALIZATION AREA
            # =====================================================================
            st.markdown("### 📊 Sentiment Confidence Graph")
            
            chart_data = pd.DataFrame({
                'Emotional Category': label_encoder.classes_,
                'Probability Score': probabilities
            })
            
            bar_chart = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('Probability Score:Q', axis=alt.Axis(format='%'), title="Confidence %"),
                y=alt.Y('Emotional Category:N', sort='-x', title="Categories"),
                color=alt.Color('Probability Score:Q', scale=alt.Scale(scheme='blues'), legend=None)
            ).properties(
                width='container',
                height=350
            )
            
            st.altair_chart(bar_chart, use_container_width=True)