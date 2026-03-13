import streamlit as st
import pandas as pd
from PIL import Image
import os
from datetime import datetime
import time
import plotly.express as px

# --- 1. DATA PERSISTENCE (BRONZE LAYER) ---
BRONZE_PATH = 'bronze_vault.csv'

def save_to_bronze(district, symptoms, neural_conf, symbolic_result):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "district": district,
        "symptoms": "|".join(symptoms),
        "neural_confidence": neural_conf,
        "symbolic_clade_prediction": symbolic_result,
        "status": "Awaiting_Human_Validation"
    }
    df = pd.DataFrame([data])
    if not os.path.isfile(BRONZE_PATH):
        df.to_csv(BRONZE_PATH, index=False)
    else:
        df.to_csv(BRONZE_PATH, mode='a', header=False, index=False)

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="Mpox Semantic Surveillance", layout="wide", page_icon="🛡️")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Project Dashboard")
    st.write("**Institution:** Makerere University")
    st.divider()
    page = st.radio("Navigate to:", ["📡 Reporting Portal", "📊 Surveillance Insights"])

# --- 4. PAGE: REPORTING PORTAL ---
if page == "📡 Reporting Portal":
    st.header("🛡️ Mpox Community Reporting Portal")
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Step 1: Live Data Capture (Bronze Layer)")
        
        # --- NEW: LIVE CAMERA INPUT ---
        input_method = st.radio("Select Input Method:", ["Live Camera Capture", "Upload Existing Image"])
        
        captured_image = None
        if input_method == "Live Camera Capture":
            captured_image = st.camera_input("Take a photo of the lesion")
        else:
            captured_image = st.file_uploader("Upload Lesion Image", type=["jpg", "png", "jpeg"])
        
        district = st.selectbox("Current Location (District)", ["Nakasongola", "Kasese", "Kampala", "Wakiso", "Mbarara", "Lwengo"])
        
        st.write("**Observed Symptoms:**")
        fever = st.checkbox("Fever")
        lymph = st.checkbox("Swollen Lymph Nodes")
        headache = st.checkbox("Headache")

        if st.button("🚀 Run Neurosymbolic Analysis"):
            if captured_image:
                symptoms_list = [s for s, val in zip(["Fever", "Lymph", "Headache"], [fever, lymph, headache]) if val]
                
                with st.status("Analyzing Live Data...", expanded=True) as status:
                    st.write("Extracting Pixels (Neural Layer)...")
                    time.sleep(1)
                    st.write("Traversing Knowledge Graph (Symbolic Layer)...")
                    time.sleep(1)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                with col2:
                    st.subheader("Step 2: Semantic Insights")
                    
                    # Simulated Logic
                    clade_prediction = "Clade Ib" if "Fever" in symptoms_list and district == "Nakasongola" else "Observation Required"
                    st.metric(label="CNN Confidence", value="92.1%")
                    
                    if clade_prediction == "Clade Ib":
                        st.error(f"⚠️ **Semantic Alert:** High correlation with {clade_prediction} patterns.")
                    else:
                        st.info("Case logged for surveillance.")

                    # Save to DWH
                    save_to_bronze(district, symptoms_list, "92.1%", clade_prediction)
                    st.toast("Data successfully captured and saved!", icon="💾")
            else:
                st.warning("Please capture or upload an image to proceed.")

# --- 5. PAGE: SURVEILLANCE INSIGHTS ---
else:
    st.header("📊 National Surveillance Insights")
    if os.path.exists(BRONZE_PATH):
        df = pd.read_csv(BRONZE_PATH)
        st.metric("Total Live Captures", len(df))
        st.write("**Real-Time Ingestion Log:**")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No live data captured yet.")