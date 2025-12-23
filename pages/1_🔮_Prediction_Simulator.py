import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_loader import load_dataset, load_prediction_model, prepare_input_data, EMITENS, IDX_QUANT, IDX_QUAL
from utils.plots import plot_interactive_forecast

st.set_page_config(page_title="Prediction Simulator", page_icon="🔮", layout="wide")
with open('style.css') as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🔮 Real-time Prediction Simulator")
st.markdown("Simulasi prediksi harga untuk **3 Hari ke Depan** berdasarkan data pasar terbaru.")

# Sidebar Controls
with st.sidebar:
    st.header("Konfigurasi")
    selected_emiten = st.selectbox("Pilih Emiten", EMITENS)
    window_size = 60 # Sesuai training

# Load Data & Models
df = load_dataset()

# Cek apakah df berhasil di-load dan tidak kosong
if df is not None and not df.empty:
    df_emiten = df[df['relevant_issuer'] == selected_emiten].sort_values('date')

    if st.button("Jalankan Prediksi", type="primary"):
        with st.spinner(f'Sedang memproses prediksi untuk {selected_emiten}...'):
            
            # 1. Prepare Data
            raw_data = prepare_input_data(df_emiten, window_size) # (60, 11)
            
            if raw_data is None:
                st.error("Data historis tidak cukup (kurang dari 60 hari).")
                st.stop()

            # 2. Load Models
            model_base, scaler = load_prediction_model(selected_emiten, 'baseline')
            model_fuse, _      = load_prediction_model(selected_emiten, 'fusion')
            
            if model_base and model_fuse:
                # 3. Scaling
                # Scaler diexpect fit dengan 11 fitur.
                data_scaled = scaler.transform(raw_data) # (60, 11)
                
                # 4. Predict Baseline (Only Quant features)
                # Input shape: (1, 60, 7)
                # PERBAIKAN: Gunakan IDX_QUANT (Huruf Besar)
                X_base = data_scaled[:, IDX_QUANT].reshape(1, window_size, len(IDX_QUANT))
                pred_base_sc = model_base.predict(X_base, verbose=0)[0] # (3,)
                
                # 5. Predict Fusion (Quant + Qual)
                # Input shape: [ (1, 60, 7), (1, 60, 4) ]
                # PERBAIKAN: Gunakan IDX_QUANT & IDX_QUAL (Huruf Besar)
                X_fuse_quant = data_scaled[:, IDX_QUANT].reshape(1, window_size, len(IDX_QUANT))
                X_fuse_qual  = data_scaled[:, IDX_QUAL].reshape(1, window_size, len(IDX_QUAL))
                pred_fuse_sc = model_fuse.predict([X_fuse_quant, X_fuse_qual], verbose=0)[0] # (3,)
                
                # 6. Inverse Scaling
                # Kita butuh dummy array untuk inverse karena scaler expect 11 fitur
                # Kita cuma mau inverse kolom index 0 (Yt)
                
                def inverse_price(pred_array, scaler):
                    # Buat dummy matrix (N, 11)
                    dummy = np.zeros((len(pred_array), 11))
                    # Isi kolom 0 dengan prediksi
                    dummy[:, 0] = pred_array
                    # Inverse
                    inv = scaler.inverse_transform(dummy)
                    return inv[:, 0] # Ambil kolom 0 saja
                
                price_base = inverse_price(pred_base_sc, scaler)
                price_fuse = inverse_price(pred_fuse_sc, scaler)
                
                # 7. Generate Dates
                last_date = df_emiten['date'].max()
                dates_fut = pd.date_range(last_date + pd.Timedelta(days=1), periods=3)
                
                # --- DISPLAY RESULTS ---
                
                # Metrics H+1
                st.subheader("Hasil Prediksi Besok (H+1)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Baseline Prediction", f"Rp {int(price_base[0]):,}", 
                              f"{price_base[0] - df_emiten['Yt'].iloc[-1]:.0f}")
                with col2:
                    st.metric("Fusion Prediction", f"Rp {int(price_fuse[0]):,}", 
                              f"{price_fuse[0] - df_emiten['Yt'].iloc[-1]:.0f}")

                # Visualization
                st.subheader("Visualisasi Proyeksi Trend")
                fig = plot_interactive_forecast(df_emiten, price_base, price_fuse, dates_fut, selected_emiten)
                st.plotly_chart(fig, use_container_width=True)
                
                # Table Detail
                st.subheader("Detail Angka (3 Hari)")
                res_df = pd.DataFrame({
                    'Tanggal': dates_fut.strftime('%d-%m-%Y'),
                    'Baseline (IDR)': price_base.astype(int),
                    'Fusion (IDR)': price_fuse.astype(int),
                    'Selisih Model': (price_base - price_fuse).astype(int)
                })
                st.table(res_df)
            else:
                st.error("Gagal memuat model. Pastikan file .h5 dan .pkl ada di folder 'models/'.")

    else:
        st.info("👈 Silakan pilih emiten di sidebar dan klik 'Jalankan Prediksi'.")
else:
    st.error("Data Frame Kosong atau Gagal Dimuat. Cek path 'data/df_fusion.csv'.")