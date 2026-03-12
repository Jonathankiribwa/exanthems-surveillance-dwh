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

# --- 2. PAGE CONFIG & BRANDING ---
st.set_page_config(page_title="Exanthems Semantic Surveillance", layout="wide", page_icon="🛡️")

# Styling for a professional academic look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004d40; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Project Dashboard")
    st.markdown("---")
    st.write("**Institution:** Makerere University")
    st.write("**Department:** College of Computing & IS")
    st.write("**Project:** Semantic-Aware DWH")
    st.divider()
    page = st.radio("Navigate to:", ["📡 Reporting Portal", "📊 Surveillance Insights"])

# --- 4. PAGE: REPORTING PORTAL ---
if page == "📡 Reporting Portal":
    st.header("🛡️ Exanthems Community Reporting Portal")
    st.info("Citizen Science Input: Capture and Classify Exanthems in Real-Time.")
    
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Step 1: Ingestion (Bronze Layer)")
        uploaded_file = st.file_uploader("Upload Lesion Image", type=["jpg", "png", "jpeg"])
        district = st.selectbox("Current Location (District)", ["Nakasongola", "Kasese", "Kampala", "Wakiso", "Mbarara", "Lwengo"])
        
        st.write("**Observed Symptoms:**")
        fever = st.checkbox("Fever")
        lymph = st.checkbox("Swollen Lymph Nodes")
        headache = st.checkbox("Headache")
        pains = st.checkbox("Muscle/Back Pains")

        if st.button("🚀 Run Neurosymbolic Analysis"):
            if uploaded_file:
                # Process Symptoms
                symptoms_list = [s for s, val in zip(["Fever", "Lymph", "Headache", "Pains"], [fever, lymph, headache, pains]) if val]
                
                with st.status("Initializing Medallion Pipeline...", expanded=True) as status:
                    st.write("Extracting Pixels (Neural Layer)...")
                    time.sleep(1)
                    st.write("Traversing Knowledge Graph (Symbolic Layer)...")
                    time.sleep(1)
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- SIMULATED RESULTS ---
                with col2:
                    st.subheader("Step 2: Semantic Insights (Gold Layer)")
                    
                    # 1. Neural Output
                    conf_score = 91.2 if "Fever" in symptoms_list else 45.6
                    st.metric(label="CNN Prediction Confidence (Mpox)", value=f"{conf_score}%")
                    
                    # 2. Symbolic Reasoning Logic
                    # (This mimics how the Knowledge Graph would reason based on the ontology)
                    clade_prediction = "Unknown"
                    if "Fever" in symptoms_list and "Lymph" in symptoms_list:
                        if district in ["Nakasongola", "Kasese"]:
                            clade_prediction = "Clade Ib (Epidemic Strain)"
                            st.error(f"⚠️ **Semantic Alert:** Patient profile strongly correlates with **{clade_prediction}** hotspots.")
                        else:
                            clade_prediction = "Clade II"
                            st.warning(f"⚠️ **Semantic Alert:** Clinical signs detected. Matching with **{clade_prediction}** patterns.")
                    else:
                        clade_prediction = "Non-Specific Exanthem"
                        st.success("✅ Semantic Reasoning: Clinical criteria for Mpox Clade Ib not fully met.")

                    # Save result
                    save_to_bronze(district, symptoms_list, f"{conf_score}%", clade_prediction)
                    st.toast("Data Saved to Bronze Vault!", icon="💾")
                    
                    st.image(Image.open(uploaded_file), caption="Uploaded Specimen", width=300)
            else:
                st.warning("Please upload an image specimen to proceed.")

# --- 5. PAGE: SURVEILLANCE INSIGHTS ---
else:
    st.header("📊 National Surveillance Insights")
    st.markdown("### Real-time BI Dashboard (Gold Layer)")

    if os.path.exists(BRONZE_PATH):
        df = pd.read_csv(BRONZE_PATH)
        
        # Dashboard Top Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Cases Logged", len(df))
        m2.metric("Active Hotspots", df['district'].nunique())
        m3.metric("Clade Ib Detections", len(df[df['symbolic_clade_prediction'].str.contains("Clade Ib", na=False)]))

        st.divider()

        # Visualizations
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Reports by District**")
            district_counts = df['district'].value_counts().reset_index()
            fig_dist = px.bar(district_counts, x='district', y='count', color='district', template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

        with c2:
            st.write("**Clade Distribution (Semantic Classification)**")
            clade_counts = df['symbolic_clade_prediction'].value_counts().reset_index()
            fig_pie = px.pie(clade_counts, names='symbolic_clade_prediction', values='count', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🗄️ Raw Bronze Layer Ingestion Log")
        st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True)
        
    else:
        st.info("No surveillance data available yet. Please use the Reporting Portal to ingest data.")

# --- FOOTER ---
st.markdown("---")
st.caption("A Semantic-Aware Data Warehouse for Crowdsourced Community-Based Surveillance. Developed by DWH-CS-1.")