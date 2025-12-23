import streamlit as st
import pandas as pd
from utils.data_loader import load_shap_data, EMITENS
from utils.plots import plot_interactive_shap

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Explainable AI - Stock Fusion",
    page_icon="🧠",
    layout="wide"
)

# 2. LOAD CSS (Supaya tema konsisten Light Mode)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 3. HEADER
st.title("🧠 Explainable AI (XAI)")
st.markdown("""
<div style='color: #4b5563; margin-bottom: 20px;'>
    <strong>Membongkar 'Black Box' Model Fusion:</strong><br>
    Modul ini memvisualisasikan bagaimana model mengambil keputusan. Apakah model lebih percaya pada 
    <em>Data Pasar (Teknikal)</em> atau <em>Berita & Sentimen (Kualitatif)</em>?
</div>
""", unsafe_allow_html=True)

st.divider()

# 4. LOAD DATA SHAP (Dari CSV, bukan Gambar!)
df_shap = load_shap_data()

if not df_shap.empty:
    # --- LAYOUT KONTROL ---
    # Kita buat layout yang bersih: Kiri untuk kontrol, Kanan untuk visualisasi
    col_ctrl, col_viz = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("⚙️ Konfigurasi")
        
        # Pilihan Mode
        view_mode = st.radio(
            "Pilih Sudut Pandang:",
            ["Global Overview (Rata-rata)", "Analisis Per Emiten"]
        )
        
        st.markdown("---")
        
        if view_mode == "Analisis Per Emiten":
            selected_emiten = st.selectbox("Pilih Saham:", EMITENS)
            
            # Tampilkan info singkat emiten
            st.info(f"""
            **Fokus Analisis:** {selected_emiten}
            
            Grafik di samping menunjukkan fitur mana yang paling 'menggerakkan' harga {selected_emiten} 
            untuk prediksi H+1.
            """)
        else:
            st.info("""
            **Fokus Analisis:** Global (8 Saham LQ45)
            
            Grafik di samping adalah rata-rata dampak fitur dari seluruh saham. 
            Ini menunjukkan 'perilaku umum' model.
            """)
            
        # Legenda Warna Manual (Biar user paham tanpa nanya)
        st.markdown("### 🏷️ Legenda Fitur")
        st.markdown("""
        <div style='display: flex; align-items: center; margin-bottom: 5px;'>
            <div style='width: 15px; height: 15px; background-color: #1f77b4; margin-right: 10px; border-radius: 3px;'></div>
            <span><strong>Teknikal</strong> (Harga, Vol, Indikator)</span>
        </div>
        <div style='display: flex; align-items: center;'>
            <div style='width: 15px; height: 15px; background-color: #d62728; margin-right: 10px; border-radius: 3px;'></div>
            <span><strong>Sentimen</strong> (Berita, Stockbit)</span>
        </div>
        """, unsafe_allow_html=True)

    with col_viz:
        # --- LOGIKA VISUALISASI ---
        if view_mode == "Global Overview (Rata-rata)":
            # Agregasi Rata-rata Global
            df_viz = df_shap.groupby(['Feature', 'Feature Name', 'Category'])['Importance'].mean().reset_index()
            title_chart = "Global Feature Importance (Rata-rata Seluruh Emiten)"
            
            # Insight Box Dinamis
            top_3 = df_viz.sort_values('Importance', ascending=False).head(3)['Feature Name'].tolist()
            st.success(f"💡 **Insight Global:** Tiga faktor penentu utama di pasar saat ini adalah **{', '.join(top_3)}**.")
            
        else:
            # Filter Per Emiten
            df_viz = df_shap[df_shap['Emiten'] == selected_emiten].copy()
            title_chart = f"Feature Importance: {selected_emiten}"
            
            # Cek apakah Sentimen masuk Top 3?
            df_sorted = df_viz.sort_values('Importance', ascending=False)
            top_3_cats = df_sorted.head(3)['Category'].tolist()
            
            if 'Sentiment' in top_3_cats:
                st.warning(f"🔥 **High Impact Sentiment:** Pada {selected_emiten}, fitur Sentimen memiliki pengaruh yang signifikan (Masuk Top 3)!")
            else:
                st.info(f"📉 **Dominasi Teknikal:** Pergerakan {selected_emiten} murni didorong oleh faktor teknikal. Sentimen belum terlalu berpengaruh.")

        # --- TAMPILKAN PLOTLY (INTERAKTIF) ---
        fig = plot_interactive_shap(df_viz, title_chart)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- DATA TABLE (EXPANDER) ---
        with st.expander("📄 Lihat Data Mentah (Tabel Angka)"):
            st.dataframe(
                df_viz.sort_values('Importance', ascending=False)[['Feature Name', 'Category', 'Importance']], 
                use_container_width=True,
                hide_index=True
            )

else:
    # Error Handling jika CSV belum diupload
    st.error("""
    ❌ **Data SHAP Tidak Ditemukan.**
    
    Pastikan Anda sudah:
    1. Menjalankan Notebook untuk generate SHAP.
    2. Mendownload `shap_values_summary.csv`.
    3. Meletakkannya di folder `data/`.
    """)