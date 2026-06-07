# app.py - Laptop Recommendation System Pro Version
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Laptop Finder Pro",
    page_icon="💻",
    layout="wide"
)

# --- CUSTOM CSS: CYBER-INDUSTRIAL DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');
    
    /* Global Base */
    .stApp {
        background-color: #050505;
        color: #e0e0e0;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #222;
    }

    /* Typography */
    h1, h2, h3, .glitch-text {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    p, span, div {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        padding: 2rem 0;
        border-bottom: 2px solid #ff0000;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1a0000 0%, #050505 100%);
    }
    
    .main-title {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        color: #ffffff;
        text-shadow: 2px 2px #ff0000;
        margin: 0;
    }

    /* Laptop Card Design */
    .laptop-card {
        background: rgba(20, 20, 20, 0.8);
        border: 1px solid #333;
        border-left: 4px solid #ff0000;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .laptop-card:hover {
        border-color: #ff0000;
        transform: translateX(10px);
        background: rgba(30, 0, 0, 0.3);
        box-shadow: -5px 0px 20px rgba(255, 0, 0, 0.2);
    }

    .model-name {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Orbitron', sans-serif;
    }

    .spec-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }

    .spec-item {
        background: #111;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #222;
    }

    .spec-label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        display: block;
    }

    .spec-value {
        font-size: 0.9rem;
        color: #fff;
        font-weight: 600;
    }

    /* Price Tag */
    .price-tag {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        color: #ff0000;
        font-weight: 700;
    }

    /* Sidebar Input Styling */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    
    /* Primary Button */
    div.stButton > button:first-child {
        background: #ff0000 !important;
        color: white !important;
        border: none !important;
        width: 100%;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0px;
        transition: 0.4s;
    }
    
    div.stButton > button:first-child:hover {
        background: #b30000 !important;
        box-shadow: 0px 0px 15px #ff0000;
    }

    /* Tooltip / Badge */
    .badge-premium {
        background: #ff0000;
        color: white;
        padding: 2px 8px;
        font-size: 0.7rem;
        border-radius: 3px;
        margin-left: 10px;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv('laptop_data.csv')

@st.cache_resource
def load_models():
    knn_model = joblib.load('laptop_recommender_model.joblib')
    scaler = joblib.load('laptop_scaler.joblib')
    label_encoders = joblib.load('laptop_label_encoders.joblib')
    return knn_model, scaler, label_encoders

@st.cache_data
def load_unique_values():
    with open('unique_values.json', 'r') as f:
        return json.load(f)

# --- CURRENCY LOGIC ---
def convert_currency(amount_inr, to_currency, exchange_rates):
    if to_currency == 'INR': return amount_inr
    rate = exchange_rates.get(to_currency, 191.5)
    return amount_inr * rate

def format_currency(amount, currency):
    if currency == 'IDR':
        return f"Rp {amount:,.0f}"
    return f"₹ {amount:,.2f}"

# Initial Load
try:
    df_clean = load_data()
    unique_vals = load_unique_values()
    exchange_rates = unique_vals.get('exchange_rates', {'IDR': 191.5})
except Exception as e:
    st.error("Missing critical files (CSV/JSON). Please check your directory.")
    st.stop()

# --- SIDEBAR DESIGN ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff0000;'>RECON UNIT</h2>", unsafe_allow_html=True)
    st.write("Configures your search parameters below:")
    
    currency_choice = st.selectbox("SYSTEM CURRENCY", ['IDR', 'INR'])
    
    budget = st.number_input(f"MAX BUDGET ({currency_choice})", 
                            min_value=0, value=15000000 if currency_choice == 'IDR' else 75000)
    
    st.markdown("---")
    ram_min = st.select_slider("MINIMUM RAM (GB)", options=[4, 8, 16, 32, 64], value=8)
    
    cpu_options = ['ANY'] + sorted(unique_vals['cpu_details'])
    cpu_detail = st.selectbox("PROCESSOR ARCHITECTURE", options=cpu_options)
    
    gpu_options = ['ANY'] + sorted(unique_vals['gpu_details'])
    gpu_detail = st.selectbox("GPU ARCHITECTURE", options=gpu_options)
    
    screen_size = st.slider("MIN DISPLAY SIZE (INCH)", 10.0, 18.0, 13.3)
    
    n_recs = st.number_input("SCAN LIMIT", 1, 20, 5)
    
    search_button = st.button("EXECUTE SEARCH")

# --- MAIN CONTENT ---
# Hero Header
st.markdown("""
    <div class="hero-container">
        <h1 class="main-title">LAPTOP FINDER <span style="color:#ff0000">PRO</span></h1>
        <p style="color: #666; margin-left: 5px;">Advanced Neural-Link Recommendation Engine v2.0</p>
    </div>
    """, unsafe_allow_html=True)

# Logic untuk Filter
def get_recommendations():
    # Konversi budget input ke INR (karena dataset biasanya dalam INR)
    budget_inr = budget / exchange_rates['IDR'] if currency_choice == 'IDR' else budget
    
    filtered = df_clean[df_clean['Price'] <= budget_inr].copy()
    
    if ram_min:
        filtered = filtered[filtered['RAM_GB'] >= ram_min]
    if cpu_detail != 'ANY':
        filtered = filtered[filtered['CPU_Detail'].str.contains(cpu_detail, case=False, na=False)]
    if gpu_detail != 'ANY':
        filtered = filtered[filtered['GPU_Detail'].str.contains(gpu_detail, case=False, na=False)]
    if screen_size:
        filtered = filtered[filtered['Inches'] >= screen_size]
        
    return filtered.sort_values('Price', ascending=False).head(n_recs)

if search_button:
    results = get_recommendations()
    
    if len(results) > 0:
        st.markdown(f"### 📡 SEARCH RESULTS: {len(results)} UNITS DETECTED")
        
        for idx, row in results.iterrows():
            price_converted = convert_currency(row['Price'], currency_choice, exchange_rates)
            
            # Card UI
            st.markdown(f"""
                <div class="laptop-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div class="model-name">{row['Model']} <span class="badge-premium">VERIFIED</span></div>
                            <div style="color: #aaa; font-size: 0.9rem;">{row['Company']} Series Hardware</div>
                        </div>
                        <div class="price-tag">{format_currency(price_converted, currency_choice)}</div>
                    </div>
                    
                    <div class="spec-grid">
                        <div class="spec-item">
                            <span class="spec-label">Processor</span>
                            <span class="spec-value">{row['CPU_Detail']}</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Graphics</span>
                            <span class="spec-value">{row['GPU_Detail']}</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Memory</span>
                            <span class="spec-value">{row['RAM_GB']}GB DDR4/LPDDR5</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Storage</span>
                            <span class="spec-value">{row['SSD_GB']}GB NVMe SSD</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">Display</span>
                            <span class="spec-value">{row['Inches']}" Panel</span>
                        </div>
                        <div class="spec-item">
                            <span class="spec-label">User Rating</span>
                            <span class="spec-value" style="color: #ff0000;">{row['Rating']}/100</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ NO UNITS FOUND. ADJUST YOUR PARAMETERS OR INCREASE BUDGET.")
else:
    # Empty State
    st.info("Sistem standby. Masukkan parameter di panel kiri dan tekan 'EXECUTE SEARCH'.")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 5rem; padding: 2rem; border-top: 1px solid #222;">
        <p style="color: #444; font-size: 0.7rem; font-family: 'Orbitron';">
            CORE OS v2.0.4 | ENCRYPTED CONNECTION | DATA SOURCE: LAPTOP_DATA.CSV
        </p>
    </div>
""", unsafe_allow_html=True)
