import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# --- IMPORT LENGKAP ---
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

# 3. HEADER & SELECTOR (DIGABUNG BIAR VAR 'selected_emiten' AMAN)
c1, c2 = st.columns([3, 1])

with c1:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <h1 style='margin:0;'>Stock Fusion AI</h1>
        <span style='background:#DCFCE7; color:#166534; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:700;'>● LIVE MARKET</span>
    </div>
    <p style='color:#6B7280; margin-top:5px; font-size:16px;'>
        Institutional-grade forecasting engine powered by <strong>Multimodal LSTM & Attention Mechanism</strong>.
    </p>
    """, unsafe_allow_html=True)

with c2:
    # Selector dipindah ke sini agar variabelnya terdefinisi sebelum dipakai filter
    selected_emiten = st.selectbox("", EMITENS, label_visibility="collapsed")

st.divider()

# 4. LOAD DATA UTAMA
with st.spinner("Connecting to Market Data Engine..."):
    df = load_dataset()

# 5. MAIN LOGIC (Sekarang aman karena 'selected_emiten' sudah ada)
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
        # Custom color logic for Delta
        st.metric("Sentiment Index", f"{sentiment_score:.3f}", f"{delta_sent:.3f}", delta_color="normal") 

    st.markdown("###")
    
    # --- MAIN TABS ---
    tab_market, tab_pred, tab_eval, tab_xai = st.tabs([
        "📈 Market Overview", 
        "🔮 Forecast Simulator", 
        "📊 Model Evaluation", 
        "🧠 Explainable AI"
    ])
    
    # =========================================
    # TAB 1: MARKET OVERVIEW (REWORKED CONTROL PANEL)
    # =========================================
    with tab_market:
        # --- CONFIGURATION TOOLBAR ---
        # Menggunakan container dengan border halus agar terlihat seperti "Toolbar"
        with st.container():
            st.markdown("""
            <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb; margin-bottom: 20px; display: flex; align-items: center; gap: 20px;">
                <span style="font-size: 12px; font-weight: 700; color: #6b7280; text-transform: uppercase;">⚙️ CHART SETTINGS</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Layout Kolom: Kiri (Date), Kanan (Indicators)
            c_tools1, c_tools2 = st.columns([1, 2])
            
            with c_tools1:
                # Date Filter
                min_date = df_e['date'].min().date()
                max_date = df_e['date'].max().date()
                default_start = max_date - timedelta(days=180)
                date_range = st.date_input("Timeframe Range", value=(default_start, max_date), min_value=min_date, max_value=max_date)

            with c_tools2:
                # Indicator Multiselect (Lebih rapi daripada banyak checkbox)
                available_inds = ["Moving Average (20)", "Volume", "MACD", "RSI"]
                default_inds = ["Volume"] # Default bersih, cuma harga & volume
                
                selected_inds = st.multiselect(
                    "Active Indicators", 
                    available_inds, 
                    default=default_inds,
                    placeholder="Add technical indicators..."
                )

        # --- CHART RENDERING ---
        # 1. Filter Data by Date
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            mask = (df_e['date'].dt.date >= start_d) & (df_e['date'].dt.date <= end_d)
            df_plot = df_e.loc[mask]
        else:
            df_plot = df_e 

        # 2. Parse Selected Indicators
        show_ma = "Moving Average (20)" in selected_inds
        show_vol = "Volume" in selected_inds
        show_macd = "MACD" in selected_inds
        show_rsi = "RSI" in selected_inds

        # 3. Plot Chart
        fig_tech = plot_advanced_technical(df_plot, selected_emiten, show_ma, show_vol, show_macd, show_rsi)
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # --- DATA GRID (Footer) ---
        st.markdown("### 📋 Historical Data Log")
        with st.expander("View Raw Data Table", expanded=False):
            display_cols = ['date', 'Yt', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
            friendly_names = {'date': 'Date', 'Yt': 'Close', 'X1': 'Open', 'X2': 'High', 'X3': 'Low', 'X4': 'Volume', 'X5': 'MACD', 'X6': 'RSI', 'X7': 'Sentiment'}
            
            df_table = df_plot[display_cols].copy().sort_values('date', ascending=False).rename(columns=friendly_names)
            df_table['Date'] = df_table['Date'].dt.strftime('%Y-%m-%d')
            
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

    # =========================================
    # TAB 2: FORECAST SIMULATOR (PROFESIONAL UI REWORK)
    # =========================================
    with tab_pred:
        # --- 1. PREDICTION CONTROL PANEL ---
        st.markdown("### 🎛️ Prediction Control Center")
        
        # Container untuk status sistem (Kesan canggih/teknis)
        with st.container():
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; text-align: center;">
                    <div style="border-right: 1px solid #e5e7eb;">
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Input Window</p>
                        <p style="margin: 5px 0 0 0; color: #111827; font-size: 18px; font-weight: 700;">60 Days</p>
                        <p style="margin: 0; color: #10b981; font-size: 11px;">● Active Lookback</p>
                    </div>
                    <div style="border-right: 1px solid #e5e7eb;">
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Forecast Horizon</p>
                        <p style="margin: 5px 0 0 0; color: #111827; font-size: 18px; font-weight: 700;">T+3 Days</p>
                        <p style="margin: 0; color: #6366f1; font-size: 11px;">● Short-term</p>
                    </div>
                    <div style="border-right: 1px solid #e5e7eb;">
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Model Engine</p>
                        <p style="margin: 5px 0 0 0; color: #111827; font-size: 18px; font-weight: 700;">Hybrid LSTM</p>
                        <p style="margin: 0; color: #f59e0b; font-size: 11px;">● Attention Mech.</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Last Data Point</p>
                        <p style="margin: 5px 0 0 0; color: #111827; font-size: 18px; font-weight: 700;">""" + df_e['date'].max().strftime('%d %b %Y') + """</p>
                        <p style="margin: 0; color: #10b981; font-size: 11px;">● Live Feed</p>
                    </div>
                </div>
            </div>
            <br>
            """, unsafe_allow_html=True)

        # Tombol Eksekusi (Full Width, Prominent)
        # Menggunakan kolom agar tombol tidak terlalu lebar di layar ultra-wide
        c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
        with c_btn2:
            run_pred = st.button("⚡ GENERATE AI FORECAST", type="primary", use_container_width=True)

        # --- 2. EXECUTION LOGIC ---
        if run_pred:
            # Tampilan loading yang lebih bersih
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            
            try:
                # A. PREPARE DATA
                my_bar.progress(10, text="Preprocessing Market Data...")
                window_size = 60
                raw_data = prepare_input_data(df_e, window_size)
                
                if raw_data is not None:
                    # B. LOAD MODELS
                    my_bar.progress(30, text=f"Loading Baseline & Fusion Models for {selected_emiten}...")
                    model_base, scaler = load_prediction_model(selected_emiten, 'baseline')
                    model_fuse, _       = load_prediction_model(selected_emiten, 'fusion')
                    
                    if model_base and model_fuse:
                        # C. SCALING & PREDICT
                        my_bar.progress(60, text="Running Inference Engine...")
                        data_scaled = scaler.transform(raw_data) 
                        
                        X_base = data_scaled[:, IDX_QUANT].reshape(1, window_size, len(IDX_QUANT))
                        pred_base_sc = model_base.predict(X_base, verbose=0)[0]
                        
                        X_fuse_quant = data_scaled[:, IDX_QUANT].reshape(1, window_size, len(IDX_QUANT))
                        X_fuse_qual  = data_scaled[:, IDX_QUAL].reshape(1, window_size, len(IDX_QUAL))
                        pred_fuse_sc = model_fuse.predict([X_fuse_quant, X_fuse_qual], verbose=0)[0]
                        
                        # D. INVERSE SCALING
                        my_bar.progress(80, text="Denormalizing Output...")
                        def inverse_price(pred_array, scaler):
                            dummy = np.zeros((len(pred_array), len(IDX_QUANT) + len(IDX_QUAL)))
                            dummy[:, 0] = pred_array 
                            inv = scaler.inverse_transform(dummy)
                            return inv[:, 0]
                        
                        price_base = inverse_price(pred_base_sc, scaler)
                        price_fuse = inverse_price(pred_fuse_sc, scaler)
                        
                        # E. GENERATE DATES
                        last_date = df_e['date'].max()
                        dates_fut = pd.date_range(last_date + timedelta(days=1), periods=3)
                        
                        # Selesai Loading
                        my_bar.progress(100, text="Completed.")
                        my_bar.empty()

                        # --- 3. RESULT DASHBOARD ---
                        st.markdown("---")
                        st.subheader("🎯 Forecast Results")

                        # A. SUMMARY CARDS (Highlight Key Numbers)
                        # Kita hitung rata-rata selisih untuk melihat sentimen
                        avg_diff = np.mean(price_fuse - price_base)
                        sentiment_signal = "Bullish Bias" if avg_diff > 0 else "Bearish Bias"
                        signal_color = "#10b981" if avg_diff > 0 else "#ef4444"

                        kpi1, kpi2, kpi3 = st.columns(3)
                        
                        # Style khusus untuk KPI Card Result
                        def kpi_card(label, value, sub, border_color="#e5e7eb"):
                            st.markdown(f"""
                            <div style="border: 1px solid {border_color}; border-radius: 10px; padding: 15px; background: white;">
                                <div style="color: #6b7280; font-size: 12px; font-weight: 600;">{label}</div>
                                <div style="color: #111827; font-size: 20px; font-weight: 700; margin-top: 5px;">{value}</div>
                                <div style="color: {border_color}; font-size: 12px; margin-top: 2px;">{sub}</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with kpi1:
                            kpi_card("Baseline Target (H+1)", f"Rp {int(price_base[0]):,}", "Conservative / Technical Only", "#2563eb") # Blue
                        with kpi2:
                            kpi_card("Fusion Target (H+1)", f"Rp {int(price_fuse[0]):,}", "Sentiment Adjusted", "#f59e0b") # Orange
                        with kpi3:
                            diff_val = int(price_fuse[0] - price_base[0])
                            sign = "+" if diff_val > 0 else ""
                            kpi_card("Sentiment Impact (Alpha)", f"{sign}Rp {diff_val:,}", sentiment_signal, signal_color)

                        # B. FAN CHART
                        st.markdown("###")
                        st.markdown("**📉 Trajectory Visualization**")
                        fig_pred = plot_interactive_forecast(df_e, price_base, price_fuse, dates_fut, selected_emiten)
                        # Tweak chart height/margin for dashboard feel
                        fig_pred.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=450)
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # C. DETAILED TABLE (Clean Look)
                        with st.expander("🔎 View Detailed Projection Table", expanded=True):
                            res_df = pd.DataFrame({
                                'Target Date': dates_fut.strftime('%d %b %Y'),
                                'Baseline Prediction': price_base,
                                'Fusion Prediction': price_fuse,
                                'Spread (Rp)': price_fuse - price_base,
                                'Spread (%)': ((price_fuse - price_base) / price_base) * 100
                            })
                            
                            st.dataframe(
                                res_df, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config={
                                    "Baseline Prediction": st.column_config.NumberColumn(format="Rp %d"),
                                    "Fusion Prediction": st.column_config.NumberColumn(format="Rp %d"),
                                    "Spread (Rp)": st.column_config.NumberColumn(format="Rp %d"),
                                    "Spread (%)": st.column_config.NumberColumn(format="%.2f%%"),
                                }
                            )
                        
                    else:
                        st.error("⚠️ Model Error: File .h5 tidak ditemukan atau rusak.")
                else:
                    st.error("⚠️ Data Error: Data historis tidak cukup untuk windowing.")
            
            except Exception as e:
                st.error(f"❌ Execution Failed: {str(e)}")

    # =========================================
    # TAB 3: EVALUATION (PROFESSIONAL AUDIT UI)
    # =========================================
    with tab_eval:
        st.markdown("### 📊 Model Performance Audit")
        df_dm, df_horizon = load_evaluation_files()
        
        if df_dm is not None and df_horizon is not None:
            # --- 1. EXECUTIVE SUMMARY (KPI CARDS) ---
            # Hitung statistik kemenangan
            baseline_wins = df_dm['Kesimpulan'].str.contains('BASELINE').sum()
            fusion_wins = df_dm['Kesimpulan'].str.contains('FUSION').sum()
            draws = df_dm['Kesimpulan'].str.contains('Seri').sum()
            
            # Tampilkan dalam Card
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #e5e7eb; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;">
                    <div>
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Baseline Wins</p>
                        <p style="margin: 5px 0 0 0; color: #2563eb; font-size: 24px; font-weight: 800;">""" + str(baseline_wins) + """</p>
                        <p style="margin: 0; color: #6b7280; font-size: 11px;">Conservative</p>
                    </div>
                    <div style="border-left: 1px solid #e5e7eb; border-right: 1px solid #e5e7eb;">
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Statistical Draws</p>
                        <p style="margin: 5px 0 0 0; color: #4b5563; font-size: 24px; font-weight: 800;">""" + str(draws) + """</p>
                        <p style="margin: 0; color: #6b7280; font-size: 11px;">No Significant Diff.</p>
                    </div>
                    <div>
                        <p style="margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;">Fusion Wins</p>
                        <p style="margin: 5px 0 0 0; color: #f59e0b; font-size: 24px; font-weight: 800;">""" + str(fusion_wins) + """</p>
                        <p style="margin: 0; color: #6b7280; font-size: 11px;">Sentiment Driven</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- 2. DIEBOLD-MARIANO TEST (STYLED TABLE) ---
            st.markdown("#### 1. Diebold-Mariano Significance Test")
            st.info("Uji statistik untuk menentukan apakah perbedaan akurasi kedua model signifikan atau hanya kebetulan (Noise).")
            
            # Coloring Logic: Hijau jika P-Value < 0.05 (Signifikan)
            def highlight_significant(val):
                color = '#dcfce7' if val < 0.05 else 'white' # Light Green
                return f'background-color: {color}'
            
            st.dataframe(
                df_dm.style.applymap(highlight_significant, subset=['P-Value']),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "DM Statistic": st.column_config.NumberColumn(format="%.4f"),
                    "P-Value": st.column_config.NumberColumn(format="%.4f"),
                    "Kesimpulan": st.column_config.TextColumn(width="medium"),
                }
            )
            
            # --- 3. HORIZON DEGRADATION (CHART + TABLE) ---
            st.markdown("---")
            st.markdown("#### 2. Horizon Degradation Analysis (Stability)")
            
            # Layout: Kiri (Chart), Kanan (Tabel Detail)
            c_h1, c_h2 = st.columns([2, 1])
            
            with c_h1:
                # Data Cleaning (Persen String -> Float)
                df_plot_h = df_horizon.copy()
                cols_mape = ['MAPE H+1', 'MAPE H+2', 'MAPE H+3']
                
                # Cek jika data berupa string "2.5%", bersihkan. Jika float, biarkan.
                for c in cols_mape:
                    if df_plot_h[c].dtype == 'object':
                        df_plot_h[c] = df_plot_h[c].astype(str).str.rstrip('%').astype('float')
                
                # Plotly Line Chart
                import plotly.graph_objects as go
                fig_h = go.Figure()
                
                # Warna-warni profesional
                colors = ['#2563eb', '#db2777', '#ca8a04', '#16a34a', '#dc2626', '#9333ea', '#0891b2', '#4b5563']
                
                for idx, row in df_plot_h.iterrows():
                    fig_h.add_trace(go.Scatter(
                        x=['H+1', 'H+2', 'H+3'],
                        y=[row['MAPE H+1'], row['MAPE H+2'], row['MAPE H+3']],
                        mode='lines+markers',
                        name=row['Emiten'],
                        line=dict(width=2, color=colors[idx % len(colors)])
                    ))
                
                fig_h.update_layout(
                    title="Error Growth Curve (MAPE)",
                    xaxis_title="Forecast Horizon",
                    yaxis_title="MAPE (%)",
                    template="plotly_white",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", y=-0.2),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_h, use_container_width=True)
                
            with c_h2:
                st.markdown("**Detailed Metrics**")
                st.dataframe(
                    df_horizon, 
                    use_container_width=True, 
                    hide_index=True
                )
                
        else:
            st.error("❌ Data Evaluasi (DM Test / Horizon) tidak ditemukan. Pastikan file CSV tersedia di folder data/.")
        
    # =========================================
    # TAB 4: EXPLAINABLE AI (MARKET DRIVERS DASHBOARD)
    # =========================================
    with tab_xai:
        st.markdown("### 🧠 Market Drivers Analysis (SHAP)")
        df_shap = load_shap_data()
        
        if not df_shap.empty:
            # --- 1. CONFIGURATION & INSIGHT PANEL ---
            c_config, c_insight = st.columns([1, 2])
            
            with c_config:
                st.markdown("""
                <div style="background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
                    <p style="color: #6b7280; font-size: 12px; font-weight: 700; margin-bottom: 10px;">ANALYSIS SCOPE</p>
                """, unsafe_allow_html=True)
                
                view_mode = st.radio("Perspective:", ["Global (All Assets)", "Single Asset Focus"], label_visibility="collapsed")
                
                if view_mode == "Single Asset Focus":
                    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer
                    # Auto-select emiten yang sedang aktif di header
                    idx_curr = EMITENS.index(selected_emiten) if selected_emiten in EMITENS else 0
                    shap_emiten = st.selectbox("Select Ticker:", EMITENS, index=idx_curr)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Legend Card
                st.markdown("""
                <div style="margin-top: 15px; background-color: white; padding: 15px; border-radius: 10px; border: 1px solid #e5e7eb;">
                    <p style="color: #6b7280; font-size: 12px; font-weight: 700; margin-bottom: 8px;">FEATURE CATEGORIES</p>
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <div style="width: 12px; height: 12px; background-color: #1f77b4; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 13px; color: #374151;"><strong>Technical</strong> (Price, Vol, Indicators)</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 12px; height: 12px; background-color: #d62728; border-radius: 3px; margin-right: 8px;"></div>
                        <span style="font-size: 13px; color: #374151;"><strong>Sentiment</strong> (News, Buzz)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c_insight:
                # DATA PREPARATION FOR VISUALIZATION
                if view_mode == "Global (All Assets)":
                    df_viz = df_shap.groupby(['Feature', 'Feature Name', 'Category'])['Importance'].mean().reset_index()
                    chart_title = "Global Market Drivers (Avg. Impact)"
                    subject = "Market (LQ45)"
                else:
                    df_viz = df_shap[df_shap['Emiten'] == shap_emiten].copy()
                    chart_title = f"Top Drivers for {shap_emiten}"
                    subject = shap_emiten

                # --- INTELLIGENT INSIGHT GENERATION ---
                # Hitung Top 3 Drivers
                df_sorted = df_viz.sort_values('Importance', ascending=False)
                top_3_names = df_sorted.head(3)['Feature Name'].tolist()
                top_3_cats = df_sorted.head(3)['Category'].tolist()
                
                # Logic Warna & Pesan
                if 'Sentiment' in top_3_cats:
                    insight_bg = "#fff7ed" # Orange Light
                    insight_border = "#f97316" # Orange
                    insight_icon = "🔥"
                    insight_title = "High Sentiment Sensitivity"
                    insight_text = f"Warning: <b>{subject}</b> is currently being driven heavily by News/Sentiment factors. Volatility is expected to be higher than technical projection."
                else:
                    insight_bg = "#eff6ff" # Blue Light
                    insight_border = "#3b82f6" # Blue
                    insight_icon = "🛡️"
                    insight_title = "Technical Structure Dominant"
                    insight_text = f"<b>{subject}</b> is behaving rationally according to technical indicators. News sentiment has minimal impact (Priced-in), making trend-following strategies safer."

                # RENDER INSIGHT CARD HTML
                st.markdown(f"""
                <div style="background-color: {insight_bg}; border-left: 5px solid {insight_border}; padding: 20px; border-radius: 8px; height: 100%;">
                    <div style="display: flex; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 24px; margin-right: 10px;">{insight_icon}</span>
                        <h3 style="margin: 0; color: #1f2937; font-size: 18px;">{insight_title}</h3>
                    </div>
                    <p style="color: #4b5563; font-size: 14px; line-height: 1.5; margin-bottom: 15px;">
                        {insight_text}
                    </p>
                    <p style="color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase; margin: 0;">
                        Top 3 Drivers: <span style="color: #111827;">{', '.join(top_3_names)}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # --- 2. MAIN VISUALIZATION ---
            st.markdown("###") # Spacer
            fig_shap = plot_interactive_shap(df_viz, chart_title)
            # Sedikit styling layout chart agar pas dengan card di atas
            fig_shap.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=450)
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # --- 3. RAW DATA (Optional but Pro) ---
            with st.expander("🔎 Audit Raw SHAP Values"):
                st.dataframe(
                    df_viz.sort_values('Importance', ascending=False)[['Feature Name', 'Category', 'Importance']], 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Importance": st.column_config.ProgressColumn(
                            "Impact Score",
                            format="%.4f",
                            min_value=0,
                            max_value=df_viz['Importance'].max(),
                        ),
                        "Category": st.column_config.TextColumn("Type"),
                    }
                )
            
        else:
            st.error("⚠️ SHAP Data Unavailable. Please run the model interpretation pipeline first.")

# --- FOOTER ---
st.markdown("<br><div style='text-align: center; color: #9ca3af; font-size: 12px;'>© 2025 Rafli Nugraha - Business Statistics ITS</div>", unsafe_allow_html=True)