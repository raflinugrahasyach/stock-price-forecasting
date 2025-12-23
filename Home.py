import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# --- IMPORT LENGKAP (Pastikan semua fungsi ter-import) ---
from utils.data_loader import (
    load_dataset, 
    load_prediction_model, 
    prepare_input_data, 
    load_shap_data, 
    load_evaluation_files, 
    EMITENS, 
    IDX_QUANT, 
    IDX_QUAL
)
from utils.plots import plot_advanced_technical, plot_interactive_forecast, plot_interactive_shap

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Stock Fusion AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. LOAD CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 3. HEADER
col_brand, col_sel = st.columns([3, 1])
with col_brand:
    st.title("Stock Fusion AI")
    st.markdown("<div style='margin-top: -15px; color: #6b7280;'>Institutional-Grade Forecasting with Multimodal Attention</div>", unsafe_allow_html=True)

with col_sel:
    selected_emiten = st.selectbox("", EMITENS, label_visibility="collapsed")

st.divider()

# 4. LOAD DATA UTAMA
df = load_dataset()

if not df.empty and selected_emiten in df['relevant_issuer'].values:
    # Filter Data Emiten & Sort
    df_e = df[df['relevant_issuer'] == selected_emiten].sort_values('date')
    
    # Ambil Data Terakhir untuk KPI
    last_row = df_e.iloc[-1]
    prev_row = df_e.iloc[-2]
    
    # KPI Metrics
    m1, m2, m3, m4 = st.columns(4)
    price_change = last_row['Yt'] - prev_row['Yt']
    pct_change = (price_change / prev_row['Yt']) * 100
    
    with m1: st.metric("Last Price", f"Rp {int(last_row['Yt']):,}", f"{pct_change:.2f}%")
    with m2: st.metric("Volume", f"{int(last_row['X4']/1000000)}M", "Shares")
    with m3: st.metric("RSI (14)", f"{last_row['X6']:.1f}", "Neutral" if 30 < last_row['X6'] < 70 else "Overbought/Sold")
    with m4:
        sentiment_score = last_row['X7']
        delta_sent = sentiment_score - prev_row['X7']
        st.metric("Sentiment Index", f"{sentiment_score:.3f}", f"{delta_sent:.3f}")

    st.markdown("###")
    
    # --- MAIN TABS ---
    tab_market, tab_pred, tab_eval, tab_xai = st.tabs([
        "📈 Market Overview", 
        "🔮 Forecast Simulator", 
        "📊 Model Evaluation", 
        "🧠 Explainable AI"
    ])
    
    # =========================================
    # TAB 1: MARKET OVERVIEW
    # =========================================
    with tab_market:
        # Date Filter
        min_date = df_e['date'].min().date()
        max_date = df_e['date'].max().date()
        default_start = max_date - timedelta(days=180)
        
        c_filter1, c_filter2 = st.columns([1, 4])
        with c_filter1:
            date_range = st.date_input("Filter Date Range", value=(default_start, max_date), min_value=min_date, max_value=max_date)
        
        # Filter Logic
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            mask = (df_e['date'].dt.date >= start_d) & (df_e['date'].dt.date <= end_d)
            df_plot = df_e.loc[mask]
        else:
            df_plot = df_e 

        st.markdown(f"**Technical Analysis Chart: {selected_emiten}**")
        fig_tech = plot_advanced_technical(df_plot, selected_emiten)
        st.plotly_chart(fig_tech, use_container_width=True)
        
        st.markdown("### Historical Data Grid")
        with st.expander("Show/Hide Data Table", expanded=False):
            display_cols = ['date', 'Yt', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
            friendly_names = {'date': 'Date', 'Yt': 'Close', 'X1': 'Open', 'X2': 'High', 'X3': 'Low', 'X4': 'Volume', 'X5': 'MACD', 'X6': 'RSI', 'X7': 'Sentiment'}
            df_table = df_plot[display_cols].copy().sort_values('date', ascending=False).rename(columns=friendly_names)
            df_table['Date'] = df_table['Date'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_table, use_container_width=True, hide_index=True)

    # =========================================
    # TAB 2: FORECAST SIMULATOR (DIPERBAIKI)
    # =========================================
    with tab_pred:
        st.markdown("### 🔮 Real-time Prediction Simulator")
        st.info(f"Simulator ini akan mengambil **60 hari data bursa terakhir** ({df_e['date'].max().date()}) untuk memprediksi harga **3 hari ke depan**.")
        
        # Tombol Eksekusi
        if st.button("🚀 Jalankan Prediksi Sekarang", type="primary"):
            with st.spinner(f'Sedang memproses model Baseline & Fusion untuk {selected_emiten}...'):
                
                # A. PREPARE DATA
                window_size = 60
                raw_data = prepare_input_data(df_e, window_size) # Mengambil 60 data terakhir
                
                if raw_data is not None:
                    # B. LOAD MODELS
                    model_base, scaler = load_prediction_model(selected_emiten, 'baseline')
                    model_fuse, _      = load_prediction_model(selected_emiten, 'fusion')
                    
                    if model_base and model_fuse:
                        # C. SCALING & PREDICT
                        data_scaled = scaler.transform(raw_data) 
                        
                        # Predict Baseline (Only Quant)
                        X_base = data_scaled[:, IDX_QUANT].reshape(1, window_size, len(IDX_QUANT))
                        pred_base_sc = model_base.predict(X_base, verbose=0)[0]
                        
                        # Predict Fusion (Quant + Qual)
                        X_fuse_quant = data_scaled[:, IDX_QUANT].reshape(1, window_size, len(IDX_QUANT))
                        X_fuse_qual  = data_scaled[:, IDX_QUAL].reshape(1, window_size, len(IDX_QUAL))
                        pred_fuse_sc = model_fuse.predict([X_fuse_quant, X_fuse_qual], verbose=0)[0]
                        
                        # D. INVERSE SCALING
                        def inverse_price(pred_array, scaler):
                            dummy = np.zeros((len(pred_array), len(IDX_QUANT) + len(IDX_QUAL)))
                            dummy[:, 0] = pred_array # Asumsi Yt di kolom 0
                            inv = scaler.inverse_transform(dummy)
                            return inv[:, 0]
                        
                        price_base = inverse_price(pred_base_sc, scaler)
                        price_fuse = inverse_price(pred_fuse_sc, scaler)
                        
                        # E. GENERATE DATES
                        last_date = df_e['date'].max()
                        dates_fut = pd.date_range(last_date + timedelta(days=1), periods=3)
                        
                        # F. DISPLAY RESULTS
                        st.divider()
                        
                        # 1. Metrics Comparison (H+1)
                        c_res1, c_res2 = st.columns(2)
                        with c_res1:
                            st.metric("Baseline Forecast (H+1)", f"Rp {int(price_base[0]):,}", f"Diff: {int(price_base[0] - last_row['Yt'])}")
                        with c_res2:
                            st.metric("Fusion Forecast (H+1)", f"Rp {int(price_fuse[0]):,}", f"Diff: {int(price_fuse[0] - last_row['Yt'])}")
                            
                        # 2. Fan Chart
                        st.subheader("Visualisasi Trend Forecast")
                        fig_pred = plot_interactive_forecast(df_e, price_base, price_fuse, dates_fut, selected_emiten)
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # 3. Table Detail
                        st.subheader("Detail Angka Prediksi (3 Hari)")
                        res_df = pd.DataFrame({
                            'Tanggal': dates_fut.strftime('%d-%m-%Y'),
                            'Baseline (IDR)': price_base.astype(int),
                            'Fusion (IDR)': price_fuse.astype(int),
                            'Selisih (Alpha)': (price_fuse - price_base).astype(int)
                        })
                        st.dataframe(res_df, use_container_width=True, hide_index=True)
                        
                    else:
                        st.error("Gagal memuat model. Pastikan file .h5 tersedia di folder models/.")
                else:
                    st.error("Data historis tidak cukup (kurang dari 60 hari).")

    # =========================================
    # TAB 3: EVALUATION
    # =========================================
    with tab_eval:
        df_dm, df_horizon = load_evaluation_files()
        
        st.subheader("1. Diebold-Mariano Significance Test")
        if df_dm is not None: st.dataframe(df_dm, use_container_width=True)
        else: st.warning("Data evaluasi DM tidak ditemukan.")
            
        st.subheader("2. Horizon Degradation Analysis (MAPE)")
        if df_horizon is not None: st.dataframe(df_horizon, use_container_width=True)
        else: st.warning("Data evaluasi Horizon tidak ditemukan.")
        
    # =========================================
    # TAB 4: EXPLAINABLE AI
    # =========================================
    with tab_xai:
        st.markdown("### 🧠 Feature Importance Analysis (SHAP)")
        df_shap = load_shap_data()
        
        if not df_shap.empty:
            c_sel1, c_sel2 = st.columns([1, 3])
            with c_sel1:
                view_mode = st.radio("Mode Analisis:", ["Global Average (All)", "Specific Issuer"])
            
            with c_sel2:
                if view_mode == "Specific Issuer":
                    shap_emiten = st.selectbox("Pilih Emiten:", EMITENS, index=EMITENS.index(selected_emiten) if selected_emiten in EMITENS else 0)
                else:
                    st.info("Menampilkan rata-rata kontribusi fitur dari seluruh 8 emiten LQ45.")

            st.divider()
            
            if view_mode == "Global Average (All)":
                df_viz = df_shap.groupby(['Feature', 'Feature Name', 'Category'])['Importance'].mean().reset_index()
                title_chart = "Global Feature Importance (Average)"
                st.success("Interpretasi: Fitur Teknikal mendominasi prediksi secara global.")
            else:
                df_viz = df_shap[df_shap['Emiten'] == shap_emiten].copy()
                title_chart = f"Feature Importance for {shap_emiten}"
                st.info(f"Analisis detail kontribusi fitur untuk {shap_emiten}.")

            fig_shap = plot_interactive_shap(df_viz, title_chart)
            st.plotly_chart(fig_shap, use_container_width=True)
            
        else:
            st.error("File 'shap_values_summary.csv' belum ditemukan.")

else:
    st.error(f"Data untuk {selected_emiten} tidak ditemukan.")

# --- FOOTER ---
st.markdown("<br><div style='text-align: center; color: #9ca3af; font-size: 12px;'>© 2025 Rafli Nugraha - Business Statistics ITS</div>", unsafe_allow_html=True)