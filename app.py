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
    page_icon="💻",
    layout="wide"
)

# ================= DESIGN CUSTOM CSS =================
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Background */
    .main {
        background-color: #f8f9fa;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #ffffff;
    }

    /* Card Styling */
    .laptop-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #007bff;
    }

    /* Price Tag */
    .price-tag {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }

    /* Spec Badge */
    .spec-badge {
        background-color: #e9ecef;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.85em;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
        color: #495057;
        border: 1px solid #dee2e6;
    }

    /* Header Styling */
    .main-title {
        background: linear-gradient(90deg, #007bff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 12px rgba(0,123,255,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# ================= END DESIGN CUSTOM CSS =============

# Title Section with Design
st.markdown('<h1 class="main-title">Laptop Finder AI 💻</h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2em; color: #6c757d;'>Temukan laptop impian berdasarkan budget dan spesifikasi terbaik untuk Anda.</p>", unsafe_allow_html=True)
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
    symbols = {'INR': '₹', 'IDR': 'Rp', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'SGD': 'S$', 'MYR': 'RM'}
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
    # Disembunyikan agar cleaner
    # st.success("Data dan model berhasil dimuat")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### ⚙️ Konfigurasi")

# Pilihan mata uang
currency = st.sidebar.selectbox(
    "Mata Uang",
    options=['IDR (Rupiah)', 'INR (Rupee)'],
    index=0
)
currency_map = {'IDR (Rupiah)': 'IDR', 'USD (Dollar)': 'USD', 'EUR (Euro)': 'EUR', 'INR (Rupee)': 'INR'}
selected_currency = currency_map[currency]

# Budget filter
st.sidebar.markdown("### 💸 Budget")
default_budget_inr = 50000
default_budget = convert_currency(default_budget_inr, 'INR', selected_currency, exchange_rates)
budget = st.sidebar.number_input(
    f"Budget Maksimal ({format_currency(0, selected_currency)[0]})",
    min_value=0.0,
    value=float(default_budget),
    step=1000000.0 if selected_currency == 'IDR' else 500.0
)

st.sidebar.markdown("### 🛠️ Spesifikasi")
# RAM filter
ram_min = st.sidebar.selectbox("RAM Minimal", options=[None, 4, 8, 16, 32], format_func=lambda x: "Semua Kapasitas" if x is None else f"{x} GB")

# CPU filter
cpu_options = ['Semua'] + unique_vals['cpu_details']
cpu_detail = st.sidebar.selectbox("Prosesor (CPU)", options=cpu_options)
cpu_detail = None if cpu_detail == 'Semua' else cpu_detail

# GPU filter
gpu_options = ['Semua'] + unique_vals['gpu_details']
gpu_detail = st.sidebar.selectbox("Kartu Grafis (GPU)", options=gpu_options)
gpu_detail = None if gpu_detail == 'Semua' else gpu_detail

# Screen size
screen_size = st.sidebar.slider("Ukuran Layar (inci)", min_value=10.0, max_value=18.0, value=13.0, step=0.1)

# Rating
rating_min = st.sidebar.slider("Rating Minimal (0-100)", 0, 100, 0, 5)

# Jumlah rekomendasi
n_recs = st.sidebar.slider("Tampilkan Hasil", 3, 10, 5)

# Search button
st.sidebar.markdown("---")
search_button = st.sidebar.button("Cari Laptop Sekarang", type="primary")

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
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 🔍 Filter Aktif")
    st.info(f"""
    💰 **Budget:** {format_currency(budget, selected_currency)}
    🧠 **RAM:** {f'{ram_min} GB' if ram_min else 'Semua'}
    🖥️ **CPU:** {cpu_detail if cpu_detail else 'Semua'}
    🎮 **GPU:** {gpu_detail if gpu_detail else 'Semua'}
    📏 **Layar:** {screen_size}"
    ⭐ **Rating:** {rating_min}+
    """)

with col2:
    if search_button:
        with st.spinner("Menganalisis database laptop..."):
            results = recommend_laptops(budget_inr, ram_min, cpu_detail, gpu_detail, screen_size, rating_min, n_recs)
            
            if len(results) > 0:
                st.markdown(f"### 🎯 Menemukan {len(results)} Laptop Terbaik")
                for idx, row in results.iterrows():
                    price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                    
                    # Custom Card UI
                    with st.container():
                        st.markdown(f"""
                        <div class="laptop-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin:0; color: #1f1f1f;">{row['Model']}</h3>
                                <span class="price-tag">{format_currency(price_conv, selected_currency)}</span>
                            </div>
                            <div style="margin-top: 15px;">
                                <span class="spec-badge">📟 {row['RAM_GB']:.0f}GB RAM</span>
                                <span class="spec-badge">💾 {row['SSD_GB']:.0f}GB SSD</span>
                                <span class="spec-badge">🖥️ {row['Inches']:.1f}" Display</span>
                                <span class="spec-badge">⭐ {row['Rating']}/100</span>
                            </div>
                            <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
                                <b>CPU:</b> {row['CPU_Detail']}<br>
                                <b>GPU:</b> {row['GPU_Detail']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Ops! Tidak ada laptop yang sesuai dengan kriteria Anda. Coba naikkan budget atau kurangi filter spesifikasi.")
    else:
        st.write("### 👋 Selamat Datang!")
        st.write("Silakan atur kriteria laptop yang Anda cari di panel sebelah kiri, lalu klik tombol **Cari Laptop Sekarang** untuk melihat rekomendasi terbaik.")
        
        # Placeholder image/icon for empty state
        st.markdown("""
        <div style="text-align: center; padding: 50px; opacity: 0.2;">
            <img src="https://cdn-icons-png.flaticon.com/512/4233/4233925.png" width="200">
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("✨ **Laptop Recommendation System v2.0**")
with footer_col2:
    st.markdown("<div style='text-align: right;'>Made with ❤️ using Streamlit & Scikit-learn</div>", unsafe_allow_html=True)
