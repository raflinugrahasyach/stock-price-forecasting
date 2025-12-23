import streamlit as st
import pandas as pd
from utils.data_loader import load_evaluation_files

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")
with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📊 Evaluasi & Validasi Model")
st.markdown("""
Halaman ini menyajikan validasi statistik kinerja model berdasarkan backtesting pada data uji.
Evaluasi dilakukan menggunakan metrik error standar dan uji signifikansi statistik.
""")

# Load Data dari File
df_dm, df_horizon = load_evaluation_files()

# --- TABEL 1: UJI DM ---
st.subheader("1. Uji Signifikansi (Diebold-Mariano Test)")
st.info("""
**Metodologi:** Uji dilakukan pada horizon **H+1** menggunakan **MSE Criterion**.
* **H0:** Tidak ada perbedaan signifikan antara akurasi Baseline dan Fusion.
* **H1:** Terdapat perbedaan signifikan.
""")

if df_dm is not None:
    # Formatting agar cantik
    st.dataframe(
        df_dm.style.format({
            'DM Statistic': '{:.4f}',
            'P-Value': '{:.4f}'
        }).applymap(lambda v: 'color: red; font-weight: bold;' if isinstance(v, str) and 'Signifikan' in v else '', subset=['Kesimpulan']),
        use_container_width=True
    )
else:
    st.warning("File 'data/tabel_dm_test.csv' tidak ditemukan.")

# --- TABEL 2: ANALISIS HORIZON ---
st.subheader("2. Analisis Degradasi Horizon (H+1 s.d H+3)")
st.markdown("""
Tabel berikut menunjukkan performa MAPE model pada berbagai horizon waktu.
Terlihat pola umum di mana **akurasi menurun (MAPE naik)** seiring bertambahnya horizon prediksi, mengonfirmasi ketidakpastian jangka panjang.
""")

if df_horizon is not None:
    st.dataframe(df_horizon, use_container_width=True)
else:
    st.warning("File 'data/df_horizon.xlsx' tidak ditemukan.")

# --- INTERPRETASI ---
st.divider()
st.success("""
**Kesimpulan Evaluasi:**
Berdasarkan Uji Diebold-Mariano dan Analisis Horizon, dapat disimpulkan bahwa **Model Baseline (Teknikal)** lebih unggul atau setara dengan Model Fusion dalam mayoritas kasus. 
Kompleksitas tambahan dari fitur sentimen tidak memberikan keuntungan statistik yang konsisten pada pasar saham LQ45.
""")