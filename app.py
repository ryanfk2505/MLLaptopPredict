# app.py - Laptop Recommendation System with Currency Converter
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
    page_icon=None,
    layout="wide"
)

# Custom CSS for gradient background and styling
st.markdown("""
<style>
    /* Main container gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main content area background */
    .main > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }
    
    /* Title styling */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj, .stSidebar {
        background: linear-gradient(180deg, #2d1b4e 0%, #1a0f2e 100%);
    }
    
    /* Sidebar text color */
    .stSidebar .stMarkdown, .stSidebar label, .stSidebar .stSelectbox label, .stSidebar .stNumberInput label {
        color: #ffffff !important;
    }
    
    /* Sidebar header */
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background: none !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f0ff 0%, #e8e0ff 100%);
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Info box styling */
    .stAlert {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 10px;
    }
    
    /* Success message styling */
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, #4caf5020 0%, #45a04920 100%);
    }
    
    /* Error message styling */
    .stAlert[data-baseweb="notification"]:has(div[data-testid="stMarkdownContainer"] p:contains("Error")) {
        background: linear-gradient(135deg, #f4433620 0%, #d32f2f20 100%);
    }
    
    /* Card/Expander content text */
    .streamlit-expanderContent {
        background: white;
        border-radius: 10px;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Column text contrast */
    .stMarkdown, p, span, div {
        color: #1a1a2e;
    }
    
    /* Sidebar selectbox and number input text */
    .stSidebar .stSelectbox div, .stSidebar .stNumberInput input {
        color: #1a1a2e !important;
        background: white !important;
    }
    
    /* Sidebar slider labels */
    .stSidebar .stSlider label {
        color: #ffffff !important;
    }
    
    /* Footer styling */
    footer {
        color: rgba(255,255,255,0.7) !important;
    }
    
    /* Column containers in main area */
    .stColumn > div {
        background: transparent;
    }
    
    /* Expander text contrast */
    .streamlit-expanderHeader p {
        color: #4a4a6a !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Laptop Recommendation System")
st.markdown("Temukan laptop terbaik sesuai budget dan kebutuhan Anda")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('laptop_data.csv')
    return df

# Load models
@st.cache_resource
def load_models():
    knn_model = joblib.load('laptop_recommender_model.joblib')
    scaler = joblib.load('laptop_scaler.joblib')
    label_encoders = joblib.load('laptop_label_encoders.joblib')
    return knn_model, scaler, label_encoders

# Load unique values
@st.cache_data
def load_unique_values():
    with open('unique_values.json', 'r') as f:
        return json.load(f)

# Fungsi konversi mata uang
def convert_currency(amount_inr, from_currency='INR', to_currency='IDR', exchange_rates=None):
    if exchange_rates is None:
        return amount_inr
    if from_currency != 'INR':
        amount_inr = amount_inr / exchange_rates[from_currency]
    if to_currency != 'INR':
        return amount_inr * exchange_rates[to_currency]
    return amount_inr

def format_currency(amount, currency):
    symbols = {'INR': 'Rp', 'IDR': 'Rp', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'SGD': 'S$', 'MYR': 'RM'}
    if currency == 'IDR':
        symbol = 'Rp'
    elif currency == 'INR':
        symbol = 'Rs'
    else:
        symbol = symbols.get(currency, '')
    
    if currency == 'IDR':
        return f"{symbol} {amount:,.0f}"
    return f"{symbol} {amount:,.2f}"

# Load semua file
try:
    df_clean = load_data()
    knn_model, scaler, label_encoders = load_models()
    unique_vals = load_unique_values()
    exchange_rates = unique_vals.get('exchange_rates', {'INR': 1, 'IDR': 191.5})
    st.success("Data dan model berhasil dimuat")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter Pencarian")

# Pilihan mata uang
currency = st.sidebar.selectbox(
    "Mata Uang",
    options=['IDR (Rupiah)', 'INR (Rupee)'],
    index=0
)
currency_map = {'IDR (Rupiah)': 'IDR', 'INR (Rupee)': 'INR'}
selected_currency = currency_map[currency]

# Budget filter (dalam currency yang dipilih)
default_budget_inr = 50000
default_budget = convert_currency(default_budget_inr, 'INR', selected_currency, exchange_rates)
budget = st.sidebar.number_input(
    f"Budget Maksimal ({format_currency(0, selected_currency)[0]})",
    min_value=0.0,
    value=float(default_budget),
    step=50000.0 if selected_currency == 'IDR' else 5.0
)

# RAM filter
ram_min = st.sidebar.selectbox("RAM Minimal (GB)", options=[None, 4, 8, 16, 32], format_func=lambda x: "Semua" if x is None else f"{x} GB")

# CPU filter
cpu_options = ['Semua'] + unique_vals['cpu_details']
cpu_detail = st.sidebar.selectbox("CPU", options=cpu_options)
cpu_detail = None if cpu_detail == 'Semua' else cpu_detail

# GPU filter
gpu_options = ['Semua'] + unique_vals['gpu_details']
gpu_detail = st.sidebar.selectbox("GPU", options=gpu_options)
gpu_detail = None if gpu_detail == 'Semua' else gpu_detail

# Screen size
screen_size = st.sidebar.slider("Layar Minimal (inci)", min_value=10.0, max_value=18.0, value=13.0, step=0.1)

# Rating
rating_min = st.sidebar.slider("Rating Minimal", 0, 100, 0, 5)

# Jumlah rekomendasi
n_recs = st.sidebar.slider("Jumlah Rekomendasi", 3, 10, 5)

# Search button
search_button = st.sidebar.button("Cari Laptop", type="primary")

# Konversi budget ke INR untuk filter
budget_inr = convert_currency(budget, selected_currency, 'INR', exchange_rates)

# Recommendation function
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

# Display
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Filter Yang Dipilih")
    st.markdown(f"**Budget:** {format_currency(budget, selected_currency)}")
    st.markdown(f"**RAM:** {f'Minimal {ram_min} GB' if ram_min else 'Semua'}")
    st.markdown(f"**CPU:** {cpu_detail if cpu_detail else 'Semua'}")
    st.markdown(f"**GPU:** {gpu_detail if gpu_detail else 'Semua'}")
    st.markdown(f"**Layar:** Minimal {screen_size}\"")
    st.markdown(f"**Rating:** Minimal {rating_min}")

with col2:
    if search_button:
        with st.spinner("Mencari laptop yang sesuai..."):
            results = recommend_laptops(budget_inr, ram_min, cpu_detail, gpu_detail, screen_size, rating_min, n_recs)
            if len(results) > 0:
                st.markdown(f"### Hasil Rekomendasi ({len(results)} laptop)")
                for idx, row in results.iterrows():
                    price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                    with st.expander(f"{row['Model'][:60]} - {format_currency(price_conv, selected_currency)}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Harga:** {format_currency(price_conv, selected_currency)}")
                            st.markdown(f"**RAM:** {row['RAM_GB']:.0f} GB")
                            st.markdown(f"**SSD:** {row['SSD_GB']:.0f} GB")
                        with c2:
                            st.markdown(f"**Layar:** {row['Inches']:.1f}\"")
                            st.markdown(f"**CPU:** {row['CPU_Detail'][:50]}")
                            st.markdown(f"**GPU:** {row['GPU_Detail'][:50]}")
            else:
                st.error("Tidak ada laptop yang sesuai dengan kriteria Anda")
    else:
        st.info("Atur filter di sidebar dan klik tombol 'Cari Laptop' untuk memulai")

# Footer
st.markdown("---")
st.markdown("Made with Streamlit & Scikit-learn")
