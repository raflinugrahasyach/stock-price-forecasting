import streamlit as st
from utils.data_loader import load_dataset, EMITENS

# Set Page Config (Wajib di baris pertama)
st.set_page_config(
    page_title="Stock Fusion AI",
    page_icon="📈",
    layout="wide"
)

# Load CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- HEADER ---
st.title("📈 Stock Price Forecasting System")
st.markdown("""
**Multimodal Fusion LSTM for LQ45 Stock Prediction** *Integrated with Sentiment Analysis (News & Social Media)*
""")
st.divider()

# --- KPI METRICS (Example using Data) ---
df = load_dataset()

if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    # Ambil data hari terakhir
    latest_date = df['date'].max()
    last_day_data = df[df['date'] == latest_date]
    
    with col1:
        st.metric("Last Data Update", latest_date.strftime('%d %b %Y'))
    
    with col2:
        top_gainer = last_day_data.sort_values('Yt', ascending=False).iloc[0]
        st.metric("Highest Price (LQ45)", f"Rp {int(top_gainer['Yt']):,}", top_gainer['relevant_issuer'])
        
    with col3:
        # Rata-rata sentimen positif hari ini
        avg_sent = last_day_data['X7'].mean()
        st.metric("Market Sentiment Index", f"{avg_sent:.2%}", "Positive")

    with col4:
        st.metric("Active Models", "Baseline & Fusion", "Ready")

# --- EXECUTIVE SUMMARY ---
st.subheader("📌 Research Overview")
st.info("""
Sistem ini membandingkan kinerja dua arsitektur Deep Learning:
1.  **Baseline Model (Unimodal):** Menggunakan data teknikal historis (OHLCV + Indikator).
2.  **Fusion Model (Multimodal):** Menggabungkan data teknikal dengan sentimen pasar (Berita & Stockbit) menggunakan mekanisme *Attention Fusion*.

**Tujuan:** Membuktikan efektivitas integrasi data kualitatif dalam peramalan harga saham jangka pendek (H+1 s.d H+3).
""")

st.subheader("💡 Cara Menggunakan Dashboard")
st.markdown("""
1.  Buka menu **Prediction Simulator** untuk melihat ramalan harga real-time.
2.  Buka menu **Model Evaluation** untuk melihat validasi statistik (MAPE, RMSE, Uji DM).
3.  Buka menu **Explainable AI** untuk melihat analisis fitur SHAP.
""")