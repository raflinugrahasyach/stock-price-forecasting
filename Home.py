import streamlit as st
import pandas as pd
from datetime import timedelta
from utils.data_loader import load_dataset, load_shap_data, load_evaluation_files, EMITENS
from utils.plots import plot_advanced_technical, plot_interactive_shap

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

# 4. LOAD DATA
df = load_dataset()

if not df.empty and selected_emiten in df['relevant_issuer'].values:
    # Filter Data Emiten
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
    
    # TABS
    tab_market, tab_pred, tab_eval, tab_xai = st.tabs([
        "📈 Market Overview", 
        "🔮 Forecast Simulator", 
        "📊 Model Evaluation", 
        "🧠 Explainable AI"
    ])
    
    # --- TAB 1: MARKET OVERVIEW (With Date Filter) ---
    with tab_market:
        # A. Date Filter Control
        min_date = df_e['date'].min().date()
        max_date = df_e['date'].max().date()
        
        # Default view: 6 bulan terakhir biar chart gak pusing
        default_start = max_date - timedelta(days=180)
        
        c_filter1, c_filter2 = st.columns([1, 4])
        with c_filter1:
            date_range = st.date_input(
                "Filter Date Range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Filter Logic
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            mask = (df_e['date'].dt.date >= start_d) & (df_e['date'].dt.date <= end_d)
            df_plot = df_e.loc[mask]
        else:
            df_plot = df_e # Fallback jika user baru klik satu tanggal

        # B. Plot Chart
        st.markdown(f"**Technical Analysis Chart: {selected_emiten}**")
        fig_tech = plot_advanced_technical(df_plot, selected_emiten)
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # C. Professional Data Table
        st.markdown("### Historical Data Grid")
        with st.expander("Show/Hide Data Table", expanded=True):
            # Siapkan kolom yang enak dibaca
            display_cols = ['date', 'Yt', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
            friendly_names = {
                'date': 'Date', 'Yt': 'Close', 'X1': 'Open', 'X2': 'High', 'X3': 'Low', 
                'X4': 'Volume', 'X5': 'MACD', 'X6': 'RSI', 'X7': 'Sentiment'
            }
            
            df_table = df_plot[display_cols].copy().sort_values('date', ascending=False)
            df_table = df_table.rename(columns=friendly_names)
            
            # Format Tanggal jadi string bersih
            df_table['Date'] = df_table['Date'].dt.strftime('%Y-%m-%d')
            
            # Tampilkan dengan fitur interaktif Streamlit (Sort & Search bawaan)
            st.dataframe(
                df_table, 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Close": st.column_config.NumberColumn(format="Rp %d"),
                    "Volume": st.column_config.NumberColumn(format="%d"),
                    "Sentiment": st.column_config.NumberColumn(format="%.4f"),
                }
            )

    # --- TAB 2, 3, 4 (Standard) ---
    with tab_pred:
        st.info("💡 Untuk simulasi prediksi Real-Time H+1 s.d H+3, silakan gunakan fitur ini.")
        st.markdown("**Panduan:** Simulator ini mengambil 60 data hari bursa terakhir dari tanggal yang tersedia untuk melakukan forecasting.")
    
    with tab_eval:
        from utils.data_loader import load_evaluation_files
        df_dm, df_horizon = load_evaluation_files()
        
        st.subheader("1. Diebold-Mariano Significance Test")
        if df_dm is not None: st.dataframe(df_dm, use_container_width=True)
        else: st.warning("Data evaluasi DM tidak ditemukan.")
            
        st.subheader("2. Horizon Degradation Analysis (MAPE)")
        if df_horizon is not None: st.dataframe(df_horizon, use_container_width=True)
        else: st.warning("Data evaluasi Horizon tidak ditemukan.")
        
    # --- TAB 4: EXPLAINABLE AI (Interactive) ---
    with tab_xai:
        st.markdown("### 🧠 Explainable AI: Feature Importance Analysis")
        st.markdown("""
        Modul ini menggunakan **SHAP (SHapley Additive exPlanations)** untuk mengukur kontribusi setiap fitur terhadap prediksi model Fusion.
        Anda dapat melihat analisis secara agregat (Global) atau spesifik per Emiten.
        """)
        
        # Load Data SHAP
        df_shap = load_shap_data()
        
        if not df_shap.empty:
            # Kontrol Interaktif
            c_sel1, c_sel2 = st.columns([1, 3])
            with c_sel1:
                view_mode = st.radio("Mode Analisis:", ["Global Average (All)", "Specific Issuer"])
            
            with c_sel2:
                if view_mode == "Specific Issuer":
                    # Dropdown pilih emiten, default ke emiten yang dipilih di sidebar
                    shap_emiten = st.selectbox("Pilih Emiten untuk Dianalisis:", EMITENS, index=EMITENS.index(selected_emiten) if selected_emiten in EMITENS else 0)
                else:
                    st.info("Menampilkan rata-rata kontribusi fitur dari seluruh 8 emiten LQ45.")

            st.divider()
            
            # Logika Filter Data
            if view_mode == "Global Average (All)":
                # Hitung rata-rata global
                df_viz = df_shap.groupby(['Feature', 'Feature Name', 'Category'])['Importance'].mean().reset_index()
                title_chart = "Global Feature Importance (Average of 8 Issuers)"
                
                # Insight Box Global
                st.success("""
                **Interpretasi Global:**
                Secara rata-rata, fitur **Teknikal (Biru)** seperti Harga Close & Open mendominasi prediksi.
                Fitur **Sentimen (Merah)** memiliki dampak yang relatif kecil, mengindikasikan bahwa model lebih mengandalkan data pasar historis daripada sinyal berita untuk prediksi H+1.
                """)
                
            else:
                # Filter per emiten
                df_viz = df_shap[df_shap['Emiten'] == shap_emiten].copy()
                title_chart = f"Feature Importance for {shap_emiten}"
                
                # Insight Spesifik (Dinamis)
                top_feature = df_viz.sort_values('Importance', ascending=False).iloc[0]
                is_sentiment_top = top_feature['Category'] == 'Sentiment'
                
                if is_sentiment_top:
                    st.warning(f"⚠️ **Temuan Menarik:** Pada {shap_emiten}, fitur Sentimen ({top_feature['Feature Name']}) menjadi faktor penentu utama!")
                else:
                    st.info(f"Pada {shap_emiten}, faktor dominan adalah **{top_feature['Feature Name']}** (Teknikal). Sentimen berperan sebagai pendukung minor.")

            # Tampilkan Chart Interaktif
            fig_shap = plot_interactive_shap(df_viz, title_chart)
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # Tampilkan Data Mentah di Expander
            with st.expander("Lihat Data Angka Detail"):
                st.dataframe(df_viz.sort_values('Importance', ascending=False), use_container_width=True)
                
        else:
            st.error("File 'shap_values_summary.csv' belum ditemukan di folder data/. Silakan export dari Notebook terlebih dahulu.")

else:
    st.error(f"Data untuk {selected_emiten} tidak ditemukan.")

# --- FOOTER ---
st.markdown("<br><div style='text-align: center; color: #9ca3af; font-size: 12px;'>© 2025 Rafli Nugraha - Business Statistics ITS</div>", unsafe_allow_html=True)