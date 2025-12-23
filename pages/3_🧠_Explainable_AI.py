import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Explainable AI", page_icon="🧠", layout="wide")
with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🧠 Explainable AI (SHAP Analysis)")
st.markdown("""
Analisis **SHAP (SHapley Additive exPlanations)** digunakan untuk menginvestigasi kontribusi setiap fitur terhadap prediksi model Fusion.
Kami membandingkan dampak **Fitur Teknikal (Biru)** vs **Fitur Sentimen (Merah)**.
""")

# --- BAGIAN 1: PERBANDINGAN SEKTORAL ---
st.header("1. Analisis Komparasi Sektoral")
st.markdown("Perbandingan rata-rata SHAP Value antara **Global**, **Banking (Non-GOTO)**, dan **Tech (GOTO)**.")

path_sectoral = "assets/shap_sectoral_comparison.png"
if os.path.exists(path_sectoral):
    st.image(path_sectoral, use_column_width=True, caption="Gambar 4.8: Perbandingan Kontribusi Fitur Antar Sektor")
    
    st.info("""
    **Interpretasi Utama:**
    1.  **Dominasi Teknikal:** Pada hampir semua sektor, fitur teknikal (seperti Harga `Yt`) mendominasi keputusan model.
    2.  **Peran Sentimen:** Fitur sentimen (Batang Merah) secara umum berada di peringkat bawah (Low Impact).
    3.  **Kasus GOTO:** Pada sektor teknologi (Panel Kanan), fitur sentimen menunjukkan relevansi yang sedikit lebih tinggi dibandingkan sektor perbankan, mengindikasikan sensitivitas terhadap berita.
    """)
else:
    st.error(f"Gambar {path_sectoral} tidak ditemukan.")

st.divider()

# --- BAGIAN 2: DETAIL PER EMITEN ---
st.header("2. Detail Granular Per Emiten")
st.markdown("Distribusi tingkat kepentingan fitur untuk setiap saham secara individu.")

path_grid = "assets/shap_per_emiten_grid.png"
if os.path.exists(path_grid):
    st.image(path_grid, use_column_width=True, caption="Detail SHAP Value untuk 8 Emiten LQ45")
else:
    st.error(f"Gambar {path_grid} tidak ditemukan.")