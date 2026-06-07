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

# --- INJEKSI CSS KUSTOM: TEMA HITAM & MERAH MINIMALIS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Font Global */
    html, body, [data-testid="stSidebar"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #000000 !important; /* Hitam pekat */
    }
    
    /* Judul Utama: Hitam dengan Efek Stroke & Aksen Merah */
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    
    /* Mengubah warna bawaan kode inline (``) agar TIDAK HIJAU */
    code {
        color: #FFFFFF !important;
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.1rem 0.4rem !important;
    }
    
    /* Kartu Expander Hasil */
    div[data-testid="stExpander"] {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 6px !important;
        margin-bottom: 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stExpander"]:hover {
        border-color: #990000 !important; /* Border merah saat hover */
    }

    /* Teks Judul Expander */
    div[data-testid="stExpander"] summary p {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    
    /* Panel Parameter Aktif */
    .filter-card {
        background: #0A0A0A;
        padding: 1.2rem;
        border-radius: 6px;
        border: 1px solid #222222;
    }
    
    .filter-card h4 {
        color: #990000 !important; /* Judul merah */
        margin-top: 0;
        border-bottom: 1px solid #222222;
        padding-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.9rem;
    }

    .filter-card p {
        color: #CCCCCC !important;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    
    /* Badge Harga Merah Gelap */
    .price-badge {
        background-color: #1A0000;
        color: #FF3333;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #440000;
    }
    
    /* Tombol Utama Streamlit (Cari Laptop) Jadi Merah */
    button[data-testid="stBaseButton-primary"] {
        background-color: #990000 !important;
        border-color: #990000 !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #CC0000 !important;
        border-color: #CC0000 !important;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #444444;
        font-size: 0.8rem;
        margin-top: 3rem;
    }

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;500&display=swap');

/* Ubah Font Global (Untuk Judul & Menu) */
html, body, [data-testid="stSidebar"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #000000 !important;
}

/* Khusus Judul Utama & Sub-judul pakai Orbitron agar sangar */
.main-title, .filter-card h4 {
    font-family: 'Orbitron', sans-serif !important;
}
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">Laptop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Temukan laptop terbaik sesuai budget dan spesifikasi kebutuhan Anda secara presisi.</p>', unsafe_allow_html=True)

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
    st.toast("Sistem siap digunakan.")
except Exception as e:
    st.error(f"Error saat memuat sistem: {e}")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### Filter Pencarian")

# Pilihan mata uang
currency = st.sidebar.selectbox(
    "Mata Uang",
    options=['IDR (Rupiah)', 'INR (Rupee)'],
    index=0
)
currency_map = {'IDR (Rupiah)': 'IDR', 'INR (Rupee)': 'INR'}
selected_currency = currency_map[currency]

# Budget filter
default_budget_inr = 50000
default_budget = convert_currency(default_budget_inr, 'INR', selected_currency, exchange_rates)
budget = st.sidebar.number_input(
    f"Budget Maksimal ({format_currency(0, selected_currency)[0].strip()})",
    min_value=0.0,
    value=float(default_budget),
    step=1000000.0 if selected_currency == 'IDR' else 500.0
)

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
screen_size = st.sidebar.slider("Ukuran Layar Minimal (inci)", min_value=10.0, max_value=18.0, value=13.0, step=0.1)

# Rating
rating_min = st.sidebar.slider("Rating Pengguna Minimal", 0, 100, 0, 5)

# Jumlah rekomendasi
n_recs = st.sidebar.slider("Jumlah Hasil Tampilan", 3, 10, 5)

st.sidebar.markdown("---")
# Search button
search_button = st.sidebar.button("Cari Laptop Terbaik", type="primary", use_container_width=True)

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

# Main Grid Display Layout
col1, col2 = st.columns([1, 2.5], gap="large")

with col1:
    st.markdown(f"""
    <div class="filter-card">
        <h4>Parameter Aktif</h4>
        <p><b>Maks Budget:</b> {format_currency(budget, selected_currency)}</p>
        <p><b>RAM Minimal:</b> {f"{ram_min} GB" if ram_min else "Semua"}</p>
        <p><b>Spesifikasi CPU:</b> {cpu_detail if cpu_detail else "Semua"}</p>
        <p><b>Spesifikasi GPU:</b> {gpu_detail if gpu_detail else "Semua"}</p>
        <p><b>Ukuran Layar:</b> ≥ {screen_size} Inci</p>
        <p><b>Rating Produk:</b> ≥ {rating_min} / 100</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if search_button:
        with st.spinner("Mencari..."):
            results = recommend_laptops(budget_inr, ram_min, cpu_detail, gpu_detail, screen_size, rating_min, n_recs)
            
            if len(results) > 0:
                st.markdown(f"### Hasil Pencarian ({len(results)} Laptop Ditemukan)")
                
                for idx, row in results.iterrows():
                    price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                    expander_title = f"{row['Model'][:55]}... — {format_currency(price_conv, selected_currency)}"
                    
                    with st.expander(expander_title):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Harga Resmi:** <span class='price-badge'>{format_currency(price_conv, selected_currency)}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Kapasitas RAM:** `{row['RAM_GB']:.0f} GB` International Standard")
                            st.markdown(f"**Penyimpanan SSD:** `{row['SSD_GB']:.0f} GB NVMe`")
                        with c2:
                            st.markdown(f"**Dimensi Layar:** `{row['Inches']:.1f}\"` Inci IPS/OLED")
                            st.markdown(f"**Central Processor:** `{row['CPU_Detail'][:50]}`")
                            st.markdown(f"**Graphics Card:** `{row['GPU_Detail'][:50]}`")
            else:
                st.error("Tidak ada laptop yang sesuai dengan kriteria filter Anda.")
    else:
        st.info("Atur preferensi Anda di panel sebelah kiri, kemudian klik tombol 'Cari Laptop Terbaik'.")

# Footer
st.markdown("""
    <hr style="border-top: 1px solid #222222; margin-top: 5rem;">
    <p class="footer-text">Powered by Streamlit Engine & Scikit-Learn KNN Algorithm • © 2026 Laptop Recommendation System</p>
""", unsafe_allow_html=True)
