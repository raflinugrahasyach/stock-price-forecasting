import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_advanced_technical(df, emiten, show_ma=True, show_vol=True, show_macd=False, show_rsi=False):
    """
    Professional Charting with Dynamic Indicator Layout (TradingView Style)
    """
    df_plot = df.copy()

    # 1. Tentukan Struktur Layout (Berapa baris?)
    panels = ['price']
    if show_vol: panels.append('volume')
    if show_macd: panels.append('macd')
    if show_rsi: panels.append('rsi')

    n_rows = len(panels)
    
    # Hitung Tinggi Baris secara Proporsional
    # Baris 1 (Harga) selalu dominan, sisanya berbagi ruang
    if n_rows == 1: row_heights = [1.0]
    elif n_rows == 2: row_heights = [0.7, 0.3]
    elif n_rows == 3: row_heights = [0.6, 0.2, 0.2]
    elif n_rows == 4: row_heights = [0.5, 0.15, 0.15, 0.2]
    else: row_heights = [1.0/n_rows] * n_rows 

    fig = make_subplots(
        rows=n_rows, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02, # Spasi antar chart rapat ala pro
        row_heights=row_heights
    )

    # --- PANEL 1: PRICE CHART (Selalu Ada) ---
    incr_color = '#00C853' # Hijau Vivid
    decr_color = '#FF3D00' # Merah Vivid
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_plot['date'],
        open=df_plot['X1'], high=df_plot['X2'],
        low=df_plot['X3'], close=df_plot['Yt'],
        name='OHLC',
        increasing_line_color=incr_color,
        decreasing_line_color=decr_color
    ), row=1, col=1)

    # Moving Average (Opsional)
    if show_ma:
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['Yt'].rolling(window=20).mean(),
            name='MA (20)', line=dict(color='#2962FF', width=1.5), opacity=0.8
        ), row=1, col=1)

    # --- PANEL DINAMIS (Volume, MACD, RSI) ---
    curr_row = 2

    # 2. VOLUME
    if show_vol:
        vol_colors = [incr_color if c >= o else decr_color for c, o in zip(df_plot['Yt'], df_plot['X1'])]
        fig.add_trace(go.Bar(
            x=df_plot['date'], y=df_plot['X4'],
            name='Volume', marker_color=vol_colors, opacity=0.5
        ), row=curr_row, col=1)
        # Format Y-Axis Volume
        fig.update_yaxes(title_text="Vol", row=curr_row, col=1, showgrid=False, showticklabels=False)
        curr_row += 1

    # 3. MACD
    if show_macd:
        # Kita pakai kolom X5 (MACD Line) dari data loader
        # Cek apakah kolom signal/hist ada di dataframe asli, jika tidak plot MACD line saja
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['X5'],
            name='MACD', line=dict(color='#2962FF', width=1.5)
        ), row=curr_row, col=1)
        
        # Coba plot signal/hist jika tersedia (tergantung load_dataset Anda)
        if 'macd_signal' in df_plot.columns:
             fig.add_trace(go.Scatter(
                x=df_plot['date'], y=df_plot['macd_signal'],
                name='Signal', line=dict(color='#FF6D00', width=1.5)
            ), row=curr_row, col=1)
        
        if 'macd_hist' in df_plot.columns:
             fig.add_trace(go.Bar(
                x=df_plot['date'], y=df_plot['macd_hist'],
                name='Hist', marker_color='#B0BEC5'
            ), row=curr_row, col=1)

        fig.update_yaxes(title_text="MACD", row=curr_row, col=1)
        curr_row += 1

    # 4. RSI
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df_plot['date'], y=df_plot['X6'],
            name='RSI', line=dict(color='#AA00FF', width=1.5)
        ), row=curr_row, col=1)
        
        # Garis Batas 30/70
        fig.add_shape(type="line", row=curr_row, col=1, xref="x", yref="y",
            x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1],
            y0=70, y1=70, line=dict(color="gray", width=1, dash="dash", opacity=0.5))
        fig.add_shape(type="line", row=curr_row, col=1, xref="x", yref="y",
            x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1],
            y0=30, y1=30, line=dict(color="gray", width=1, dash="dash", opacity=0.5))
            
        fig.update_yaxes(title_text="RSI", range=[0,100], row=curr_row, col=1)
        curr_row += 1

    # STYLING GLOBAL
    height_calc = 400 + (n_rows * 100) # Tinggi otomatis menyesuaikan jumlah panel
    
    fig.update_layout(
        title=dict(text=f"<b>{emiten}</b> Market Action", font=dict(size=18, family="Inter")),
        template="plotly_white",
        height=height_calc,
        showlegend=False,
        margin=dict(l=10, r=40, t=50, b=20),
        hovermode="x unified",
        xaxis=dict(showgrid=False, type="date", rangeslider=dict(visible=False))
    )
    
    # Hilangkan label X-axis di chart bagian atas agar tidak menumpuk
    for i in range(1, n_rows):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    return fig

def plot_interactive_forecast(df_hist, pred_base, pred_fuse, dates_fut, emiten):
    """
    Fan Chart untuk Halaman Prediksi
    """
    # Ambil data secukupnya untuk konteks visual (misal 3 bulan terakhir)
    last_30 = df_hist.tail(90)
    
    fig = go.Figure()
    
    # Historis
    fig.add_trace(go.Scatter(
        x=last_30['date'], y=last_30['Yt'],
        mode='lines', name='Historical',
        line=dict(color='#111827', width=2)
    ))
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=dates_fut, y=pred_base,
        mode='lines+markers', name='Baseline Forecast',
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(symbol='circle')
    ))
    
    # Fusion
    fig.add_trace(go.Scatter(
        x=dates_fut, y=pred_fuse,
        mode='lines+markers', name='Fusion Forecast',
        line=dict(color='#0052CC', width=3),
        marker=dict(symbol='diamond', size=8)
    ))
    
    # Connector line
    fig.add_trace(go.Scatter(
        x=[last_30['date'].iloc[-1], dates_fut[0]],
        y=[last_30['Yt'].iloc[-1], pred_fuse[0]],
        mode='lines', showlegend=False,
        line=dict(color='gray', width=1, dash='dot')
    ))

    fig.update_layout(
        title=f"Forecast Scenario: {emiten} (Next 3 Days)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
    )
    return fig

def plot_interactive_shap(df_shap, title_text):
    """
    Plot SHAP Values secara Interaktif (Bar Chart Horizontal).
    Warna otomatis beda antara Technical vs Sentiment.
    """
    # Sort biar yang paling penting di atas
    df_sorted = df_shap.sort_values('Importance', ascending=True)
    
    # Warna: Biru (Tech), Merah (Sentiment)
    colors = ['#d62728' if cat == 'Sentiment' else '#1f77b4' for cat in df_sorted['Category']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['Feature Name'], # Pakai nama yang cantik
        x=df_sorted['Importance'],
        orientation='h',
        marker=dict(color=colors, opacity=0.9),
        text=df_sorted['Importance'].apply(lambda x: f"{x:.4f}"), # Tampilkan angka di bar
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.5f}<br>Category: %{customdata}<extra></extra>',
        customdata=df_sorted['Category']
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=18)),
        xaxis_title="Mean |SHAP Value| (Impact Magnitude)",
        yaxis_title=None,
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )
    
    # Tambahkan Legend Manual pakai Annotation dummy (biar user tahu merah itu apa)
    fig.add_annotation(x=1, y=0, xref='paper', yref='paper', text='🟦 Technical', showarrow=False, xanchor='right', yanchor='bottom', yshift=-30, xshift=-80)
    fig.add_annotation(x=1, y=0, xref='paper', yref='paper', text='🟥 Sentiment', showarrow=False, xanchor='right', yanchor='bottom', yshift=-30)

    return fig