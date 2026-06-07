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

# --- INJEKSI CSS KUSTOM YANG MENDUKUNG DARK & LIGHT MODE ---
st.markdown("""
    <style>
    /* Mengimpor Google Font Premium */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Mengubah Font Global */
    html, body, [data-testid="stSidebar"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Styling Judul Utama dengan Gradasi Lebih Tajam & Terlihat di Dark Mode */
    .main-title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #3B82F6 0%, #10B981 50%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: inline-block;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 2rem;
    }
    
    /* Styling Kartu Hasil Pencarian (Expander) agar adaptif di Dark Mode */
    div[data-testid="stExpander"] {
        background-color: #1E293B !important; /* Warna latar gelap elegan */
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stExpander"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4) !important;
        border-color: #3B82F6 !important;
    }

    /* Memaksa teks judul expander agar berwarna cerah dan kontras */
    div[data-testid="stExpander"] summary p {
        color: #F8FAFC !important;
        font-weight: 600 !important;
    }
    
    /* Panel Ringkasan Filter / Parameter Aktif */
    .filter-card {
        background: #1E293B;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #334155;
        color: #F8FAFC !important;
    }
    
    .filter-card h4 {
        color: #3B82F6 !important;
        margin-top: 0;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.5rem;
        font-weight: 700;
    }

    .filter-card p {
        color: #CBD5E1 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Badge Harga Premium */
    .price-badge {
        background-color: #1E1B4B;
        color: #818CF8;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid #312E81;
    }
    
    /* Footer Modis */
    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header dengan komponen HTML kustom
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
    st.toast("Data dan model kecerdasan buatan berhasil dimuat!")
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

# Budget filter (dalam currency yang dipilih)
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
        with st.spinner("Menganalisis database dan mencocokkan spesifikasi..."):
            results = recommend_laptops(budget_inr, ram_min, cpu_detail, gpu_detail, screen_size, rating_min, n_recs)
            
            if len(results) > 0:
                st.markdown(f"### Hasil Pencarian ({len(results)} Laptop Ditemukan)")
                
                for idx, row in results.iterrows():
                    price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                    
                    # Menggabungkan teks nama dan harga untuk judul expander
                    expander_title = f"{row['Model'][:55]}... — {format_currency(price_conv, selected_currency)}"
                    
                    with st.expander(expander_title):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Harga Resmi:** <span class='price-badge'>{format_currency(price_conv, selected_currency)}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Kapasitas RAM:** `{row['RAM_GB']:.0f} GB` International Standard")
                            st.markdown(f"**Penyimpanan SSD:** `{row['SSD_GB']:.0f} GB NVMe` ")
                        with c2:
                            st.markdown(f"**Dimensi Layar:** `{row['Inches']:.1f}\"` Inci IPS/OLED")
                            st.markdown(f"**Central Processor:** `{row['CPU_Detail'][:50]}`")
                            st.markdown(f"**Graphics Card:** `{row['GPU_Detail'][:50]}`")
            else:
                st.error("Kombinasi filter terlalu ketat. Tidak ada laptop yang sesuai dengan kriteria dan budget Anda. Silakan naikkan budget atau kurangi filter spesifikasi.")
    else:
        st.info("Atur preferensi Anda di panel sebelah kiri, kemudian klik tombol 'Cari Laptop Terbaik' untuk memulai komparasi data.")

# Footer dengan gaya kustom minimalis
st.markdown("""
    <hr style="border-top: 1px solid #334155; margin-top: 5rem;">
    <p class="footer-text">Powered by <b>Streamlit Engine</b> & <b>Scikit-Learn KNN Algorithm</b> • © 2026 Laptop Recommendation System</p>
""", unsafe_allow_html=True)
