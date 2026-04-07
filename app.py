import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="SeDW - Exanthem Surveillance Uganda",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Persistence Logic (Bronze Layer) ---
DB_FILE = "sedw_bronze_vault.csv"

def init_db():
    """Initializes the CSV database if it doesn't exist."""
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["timestamp", "district", "role", "symptoms", "trust_score", "k_anonymity"])
        df.to_csv(DB_FILE, index=False)

def save_observation(district, role, symptoms, trust_score, k_anonymity):
    """Saves a new observation to the Bronze Layer."""
    new_data = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "district": district,
        "role": role,
        "symptoms": "|".join(symptoms),
        "trust_score": trust_score,
        "k_anonymity": k_anonymity
    }])
    new_data.to_csv(DB_FILE, mode='a', header=False, index=False)

init_db()

# --- Mock Semantic Backend Logic ---
def get_semantic_trust_score(metadata):
    """Simulates the Epistemic Trust Module (Layer 1 & 5)"""
    score = 0.4
    if metadata['location_verified']: score += 0.2
    if len(metadata['clinical_context']) >= 2: score += 0.2
    if metadata['observer_role'] in ["Community Health Worker", "Clinician"]: score += 0.15
    return min(score, 1.0)

# --- UI Header ---
st.markdown("""
    <div style='background-color: #004d40; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin:0;'>🛡️ Semantic-Aware Data Warehouse (SeDW)</h1>
        <p style='color: #e0f2f1; margin:0;'>L6: Mobile-Edge Ingestion & Epidemiological Intelligence Portal</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://seeklogo.com/vector-logo/550310/makerere-university-mak-uganda", width=120)
    st.markdown("### System Navigation")
    page = st.radio("", ["📊 Surveillance Dashboard", "📸 Live Data Capture", "🕸️ Knowledge Graph View"])
    st.divider()
    st.caption("DWH-CS-1 Project")
    st.caption("Active Mode: Neuro-symbolic pipeline")

# --- PAGE 1: Surveillance Dashboard ---
if page == "📊 Surveillance Dashboard":
    st.markdown("### Epidemiological Intelligence (Gold Layer)")
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Observations", "1,284", "+12% this week")
    col2.metric("High-Trust Alerts (>0.8)", "42", "Action Required", delta_color="inverse")
    col3.metric("OOD Robustness", "94.2%", "Model Confidence")
    col4.metric("Active Hotspots", "4", "Districts")

    st.divider()

    # Layout for Charts
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.markdown("#### 📈 Temporal Transmission Trends")
        # Generate realistic-looking mock data
        dates = pd.date_range(start='2026-03-01', periods=14)
        trend_df = pd.DataFrame({
            'Date': dates,
            'Mpox (Clade Ib)': np.random.randint(5, 25, size=14) + np.arange(14),
            'Varicella': np.random.randint(15, 30, size=14) - np.arange(14)*0.5
        })
        fig_line = px.line(trend_df, x='Date', y=['Mpox (Clade Ib)', 'Varicella'], 
                           color_discrete_sequence=['#d32f2f', '#1976d2'])
        fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", legend_title_text='Classification')
        st.plotly_chart(fig_line, use_container_width=True)

    with chart_col2:
        st.markdown("#### 🗺️ Spatial Distribution")
        # Mock geospatial data around Central Uganda
        geo_data = pd.DataFrame({
            'lat': [0.3476, 0.3136, 1.3000, 0.0500],
            'lon': [32.5825, 32.5811, 32.4000, 32.4600],
            'cases': [45, 30, 85, 20],
            'district': ['Kampala', 'Wakiso', 'Nakasongola', 'Mpigi']
        })
        fig_map = px.scatter_mapbox(geo_data, lat="lat", lon="lon", size="cases", 
                                    color="cases", hover_name="district", 
                                    color_continuous_scale=px.colors.sequential.Reds, size_max=25, zoom=6)
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

# --- PAGE 2: Live Data Capture ---
elif page == "📸 Live Data Capture":
    st.markdown("### L6: Decentralized Mobile-Edge Ingestion")
    st.info("Upload multimodal citizen science data. Metadata is semantically enriched upon ingestion.")

    col_img, col_meta = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 1. Visual Evidence")
        source = st.radio("Input Source:", ["Camera Capture", "File Upload"], horizontal=True)
        img_file = st.camera_input("Scan Exanthem") if source == "Camera Capture" else st.file_uploader("Upload Image", type=['jpg', 'png'])
        
        if img_file:
            st.success("✅ High-resolution artifact buffered.")

    with col_meta:
        st.markdown("#### 2. Semantic Context & Governance")
        role = st.selectbox("Observer Role (Epistemic Source)", ["Citizen Scientist", "Community Health Worker", "Clinician"])
        district = st.selectbox("Geographic Node", ["Kampala", "Wakiso", "Nakasongola", "Mbarara", "Kasese"])
        symptoms = st.multiselect("Clinical Signs (Ontology Mapping)", ["Fever", "Lymphadenopathy", "Headache", "Myalgia", "Asthenia"])
        
        st.markdown("##### Privacy Constraints (L5)")
        k_anon = st.toggle("Enable Spatial k-Anonymity", value=True, help="Obfuscates exact GPS to protect patient identity.")

        if st.button("🚀 Execute Neuro-symbolic Pipeline", use_container_width=True, type="primary"):
            if not img_file:
                st.error("Please provide visual evidence to proceed.")
            else:
                with st.status("Executing 8-Layer Locator Pipeline...", expanded=True) as status:
                    st.write("📥 L3: Ingesting multimodal data into Bronze Vault...")
                    time.sleep(1)
                    st.write("🧠 L7: Running CNN visual feature extraction...")
                    time.sleep(1.5)
                    st.write("🔗 L2: Aligning metadata with ExanthemObservation Ontology...")
                    
                    trust = get_semantic_trust_score({
                        'location_verified': k_anon,
                        'clinical_context': symptoms,
                        'observer_role': role
                    })
                    time.sleep(1)
                    
                    st.write(f"🛡️ L5: Epistemic Trust Score Calculated: **{trust:.2f}**")
                    
                    if trust >= 0.7:
                        st.write("✅ **Validation Passed:** Promoting to Silver Knowledge Graph.")
                        save_observation(district, role, symptoms, trust, k_anon)
                        status.update(label="Semantic Ingestion Complete!", state="complete", expanded=False)
                        st.balloons()
                    else:
                        st.write("⚠️ **Validation Warning:** Trust score too low. Flagged for human-in-the-loop review.")
                        status.update(label="Ingestion Paused: Awaiting Review", state="error", expanded=False)

# --- PAGE 3: Knowledge Graph View ---
elif page == "🕸️ Knowledge Graph View":
    st.markdown("### Semantic Traceability Interface")
    st.write("Unlike traditional BI, this interface allows public health officials to query the exact provenance and semantic relationships of flagged anomalies.")
    
    st.code("""
    PREFIX ex: <http://exanthem.org/ontology#>
    SELECT ?observation ?trust ?diagnosis ?symptoms
    WHERE {
        ?observation ex:hasTrustScore ?trust .
        ?observation ex:locatedIn ?district .
        ?observation ex:exhibits ?symptoms .
        FILTER (?trust > 0.8 && ?district = "Nakasongola")
    }
    """, language="sql")
    
    st.markdown("#### 🗄️ Raw Bronze Vault Data (Local)")
    try:
        db_df = pd.read_csv(DB_FILE)
        if db_df.empty:
            st.info("The local database is currently empty. Go to the Capture tab to ingest data.")
        else:
            st.dataframe(db_df.tail(10), use_container_width=True)
            
            # Allow examiners to download the CSV to prove it works
            csv = db_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Raw Ingestion Log", data=csv, file_name="bronze_layer_export.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("Database file not found.")