# app.py - Laptop Recommendation System with Premium Dark Theme
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Laptop Recommendation System",
    page_icon="💻",
    layout="wide"
)

# ================= PREMIUM DARK RED DESIGN =================
st.markdown("""
    <style>
    /* Mengubah font global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Background Utama & Sidebar */
    .stApp {
        background-color: #0e0e0e;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161616;
        border-right: 1px solid #330000;
    }

    /* Header Styling */
    .main-title {
        color: #ff4b4b;
        font-weight: 800;
        font-size: 3.5rem;
        letter-spacing: -1px;
        margin-bottom: 0px;
        text-transform: uppercase;
    }
    
    .sub-title {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Card Laptop Premium */
    .laptop-card {
        background-color: #1a1a1a;
        padding: 24px;
        border-radius: 4px; /* Sudut lebih tegas, bukan bulat */
        border: 1px solid #333;
        border-left: 4px solid #ff4b4b; /* Aksen merah tegas */
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .laptop-card:hover {
        background-color: #222;
        border-color: #ff4b4b;
        transform: translateY(-2px);
    }

    /* Price & Badges */
    .price-text {
        color: #ff4b4b;
        font-weight: 700;
        font-size: 1.4rem;
    }
    
    .spec-label {
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .spec-value {
        color: #eee;
        font-weight: 600;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 4px;
        width: 100%;
        font-weight: 700;
        padding: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background-color: #cc0000;
        border: none;
        color: white;
    }

    /* Menghilangkan elemen biru bawaan streamlit (fokus permintaan user) */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #ff4b4b;
    }
    .stSlider [data-baseweb="slider"] [aria-valuemax] {
        background-color: #ff4b4b;
    }
    
    /* Input & Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #1a1a1a;
        border-color: #333;
    }
    
    /* Divider */
    hr {
        border-color: #330000;
    }
    </style>
    """, unsafe_allow_html=True)

# ================= TITLE SECTION =================
st.markdown('<h1 class="main-title">LAPTOP FINDER.</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">High-Performance Recommendation System Engine</p>', unsafe_allow_html=True)

# Load data (Logic Tetap)
@st.cache_data
def load_data():
    df = pd.read_csv('laptop_data.csv')
    return df

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

# Fungsi konversi mata uang (Logic Tetap)
def convert_currency(amount_inr, from_currency='INR', to_currency='IDR', exchange_rates=None):
    if exchange_rates is None:
        return amount_inr
    if from_currency != 'INR':
        amount_inr = amount_inr / exchange_rates[from_currency]
    if to_currency != 'INR':
        return amount_inr * exchange_rates[to_currency]
    return amount_inr

def format_currency(amount, currency):
    symbols = {'INR': '₹', 'IDR': 'Rp', 'USD': '$'}
    symbol = symbols.get(currency, '')
    if currency == 'IDR':
        return f"{symbol} {amount:,.0f}"
    return f"{symbol} {amount:,.2f}"

# Load Files
try:
    df_clean = load_data()
    knn_model, scaler, label_encoders = load_models()
    unique_vals = load_unique_values()
    exchange_rates = unique_vals.get('exchange_rates', {'INR': 1, 'IDR': 191.5})
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Sidebar filters (UI Update)
st.sidebar.markdown("<h2 style='color: #ff4b4b;'>CONFIG</h2>", unsafe_allow_html=True)

currency = st.sidebar.selectbox(
    "CURRENCY",
    options=['IDR (Rupiah)', 'INR (Rupee)'],
    index=0
)
currency_map = {'IDR (Rupiah)': 'IDR', 'INR (Rupee)': 'INR'}
selected_currency = currency_map[currency]

default_budget_inr = 50000
default_budget = convert_currency(default_budget_inr, 'INR', selected_currency, exchange_rates)
budget = st.sidebar.number_input(
    f"MAX BUDGET ({selected_currency})",
    min_value=0.0,
    value=float(default_budget),
    step=1000000.0 if selected_currency == 'IDR' else 5000.0
)

st.sidebar.markdown("---")
ram_min = st.sidebar.selectbox("MIN RAM", options=[None, 4, 8, 16, 32], format_func=lambda x: "Any" if x is None else f"{x} GB")

cpu_options = ['Semua'] + unique_vals['cpu_details']
cpu_detail = st.sidebar.selectbox("CPU TYPE", options=cpu_options)
cpu_detail = None if cpu_detail == 'Semua' else cpu_detail

gpu_options = ['Semua'] + unique_vals['gpu_details']
gpu_detail = st.sidebar.selectbox("GPU TYPE", options=gpu_options)
gpu_detail = None if gpu_detail == 'Semua' else gpu_detail

screen_size = st.sidebar.slider("SCREEN SIZE (INCH)", min_value=10.0, max_value=18.0, value=13.0, step=0.1)
rating_min = st.sidebar.slider("MIN RATING", 0, 100, 0)
n_recs = st.sidebar.slider("RESULTS", 3, 10, 5)

search_button = st.sidebar.button("SEARCH LAPTOP")

# Recommendation function (Logic Tetap)
budget_inr = convert_currency(budget, selected_currency, 'INR', exchange_rates)

def recommend_laptops(price_max_inr, ram_min=None, cpu_detail=None, gpu_detail=None, screen_size_min=None, rating_min=None, n=5):
    filtered = df_clean[df_clean['Price'] <= price_max_inr].copy()
    if ram_min:
        filtered = filtered[filtered['RAM_GB'] >= ram_min]
    if cpu_detail:
        filtered = filtered[filtered['CPU_Detail'].str.contains(cpu_detail, case=False, na=False)]
    if gpu_detail:
        filtered = filtered[filtered['GPU_Detail'].str.contains(gpu_detail, case=False, na=False)]
    if screen_size_min:
        filtered = filtered[filtered['Inches'] >= screen_size_min]
    if rating_min:
        filtered = filtered[filtered['Rating'] >= rating_min]
    if len(filtered) == 0:
        return pd.DataFrame()
    return filtered.sort_values('Price').head(n).reset_index(drop=True)

# Main Display
col_info, col_res = st.columns([1, 3])

with col_info:
    st.markdown("<h4 style='color: #ff4b4b;'>ACTIVE FILTERS</h4>", unsafe_allow_html=True)
    st.write(f"Budget: **{format_currency(budget, selected_currency)}**")
    st.write(f"RAM: **{ram_min if ram_min else 'Any'}**")
    st.write(f"CPU: **{cpu_detail if cpu_detail else 'Any'}**")
    st.write(f"Rating: **{rating_min}+**")

with col_res:
    if search_button:
        with st.spinner("Processing Data..."):
            results = recommend_laptops(budget_inr, ram_min, cpu_detail, gpu_detail, screen_size, rating_min, n_recs)
            if len(results) > 0:
                for idx, row in results.iterrows():
                    price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                    
                    # HTML Card Design (Premium Black & Red)
                    st.markdown(f"""
                    <div class="laptop-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <h3 style="margin:0; color: #fff; font-size: 1.2rem;">{row['Model']}</h3>
                                <p style="margin: 5px 0 15px 0; color: #ff4b4b; font-weight: bold;">{row['Company']}</p>
                            </div>
                            <div class="price-text">{format_currency(price_conv, selected_currency)}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                            <div>
                                <div class="spec-label">RAM / SSD</div>
                                <div class="spec-value">{row['RAM_GB']:.0f}GB / {row['SSD_GB']:.0f}GB</div>
                            </div>
                            <div>
                                <div class="spec-label">Display</div>
                                <div class="spec-value">{row['Inches']:.1f}" Screen</div>
                            </div>
                            <div>
                                <div class="spec-label">Engine Rating</div>
                                <div class="spec-value">⭐ {row['Rating']}/100</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #333; font-size: 0.85rem;">
                            <span style="color: #888;">CPU:</span> <span style="color: #ccc;">{row['CPU_Detail']}</span><br>
                            <span style="color: #888;">GPU:</span> <span style="color: #ccc;">{row['GPU_Detail']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("No laptops found matching your criteria.")
    else:
        st.markdown("""
        <div style="padding: 100px; text-align: center; border: 1px dashed #333;">
            <h2 style="color: #333;">WAITING FOR INPUT</h2>
            <p style="color: #555;">Adjust filters on the sidebar and click 'Search Laptop'</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #333; font-size: 0.8rem; letter-spacing: 2px;'>SYSTEM CORE v2.0 | BUILT WITH STREAMLIT</div>", unsafe_allow_html=True)
