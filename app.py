import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model

# Load model
model = load_model('final_model')

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")

st.title("🌾 Smart Crop Recommendation System")

st.markdown("Enter soil and weather conditions")

# Inputs
N = st.number_input("Nitrogen", 0.0, 200.0, 50.0)
P = st.number_input("Phosphorus", 0.0, 200.0, 50.0)
K = st.number_input("Potassium", 0.0, 200.0, 50.0)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall", 0.0, 500.0, 100.0)

if st.button("🔍 Recommend Crop"):

    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
        columns=['N','P','K','temperature','humidity','ph','rainfall'])

    # Predict probabilities
    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    # Top 3
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_crops = [classes[i] for i in top3_idx]
    top3_probs = probs[top3_idx]

    st.subheader("🌱 Top 3 Crops")

    for i in range(3):
        st.success(f"{i+1}. {top3_crops[i]} ({top3_probs[i]*100:.2f}%)")

    # Chart
    chart_df = pd.DataFrame({
        'Crop': top3_crops,
        'Probability': top3_probs * 100
    })

    st.bar_chart(chart_df.set_index('Crop'))

    # Soil analysis
    st.subheader("🧪 Soil Health")

    issues = []
    actions = []

    if N < 50:
        issues.append("Low Nitrogen")
        actions.append("Apply urea or compost")
    elif N > 120:
        issues.append("High Nitrogen")
        actions.append("Reduce fertilizer")

    if P < 40:
        issues.append("Low Phosphorus")
        actions.append("Add DAP")
    elif P > 100:
        issues.append("High Phosphorus")
        actions.append("Avoid excess fertilizer")

    if K < 40:
        issues.append("Low Potassium")
        actions.append("Apply potash")
    elif K > 80:
        issues.append("High Potassium")
        actions.append("Reduce potassium")

    if ph < 5.5:
        issues.append("Acidic Soil")
        actions.append("Apply lime")
    elif ph > 7.5:
        issues.append("Alkaline Soil")
        actions.append("Add organic matter")

    if len(issues) == 0:
        st.success("✅ Soil is optimal")
    else:
        st.warning("⚠️ Soil needs improvement")

        for i in range(len(issues)):
            st.write(f"- {issues[i]} → {actions[i]}")

    st.subheader("🌾 Final Recommendation")
    st.success(f"Best Crop: {top3_crops[0]}")