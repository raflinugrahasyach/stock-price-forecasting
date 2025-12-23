import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_advanced_technical(df, emiten):
    """
    Advanced Charting: Candlestick + Volume + MACD/RSI
    """
    # PERBAIKAN: Gunakan seluruh data yang dikirim dari Home.py
    # Jangan dipotong .tail(150) lagi!
    df_plot = df.copy()
    
    # Create Subplots: 3 Rows
    # Row 1: Candlestick (Main) - 55% height
    # Row 2: Volume - 20% height
    # Row 3: MACD/RSI - 25% height
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03,
        row_heights=[0.55, 0.20, 0.25],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # --- 1. CANDLESTICK (Main) ---
    fig.add_trace(go.Candlestick(
        x=df_plot['date'],
        open=df_plot['X1'],   # Open
        high=df_plot['X2'],   # High
        low=df_plot['X3'],    # Low
        close=df_plot['Yt'],  # Close
        name='Price',
        increasing_line_color='#00C076', # Growth Green
        decreasing_line_color='#FF4B4B'  # Loss Red
    ), row=1, col=1)

    # --- 2. VOLUME (Bar) ---
    # Warna volume ikut trend harga
    colors = ['#00C076' if row['Yt'] >= row['X1'] else '#FF4B4B' for index, row in df_plot.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df_plot['date'],
        y=df_plot['X4'], # Volume
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)

    # --- 3. TECHNICALS (MACD & RSI) ---
    # MACD (Blue) - Primary Y
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['X5'], 
        name='MACD', line=dict(color='#0052CC', width=1.5)
    ), row=3, col=1, secondary_y=False)
    
    # RSI (Purple) - Secondary Y Axis
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['X6'], 
        name='RSI', line=dict(color='#8833FF', width=1.5, dash='dot')
    ), row=3, col=1, secondary_y=True)

    # Garis Batas RSI 30/70
    fig.add_shape(type="line", row=3, col=1, xref="x", yref="y2",
                  x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1],
                  y0=70, y1=70, line=dict(color="gray", width=1, dash="dash"))
    fig.add_shape(type="line", row=3, col=1, xref="x", yref="y2",
                  x0=df_plot['date'].iloc[0], x1=df_plot['date'].iloc[-1],
                  y0=30, y1=30, line=dict(color="gray", width=1, dash="dash"))

    # --- LAYOUT STYLING ---
    fig.update_layout(
        title=dict(text=f"<b>{emiten}</b> Market Overview", font=dict(size=20)),
        template="plotly_white",
        height=800, # Agak tinggi biar jelas
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        
        # Rangeslider PENTING: Agar user bisa zoom in/out manual
        xaxis=dict(
            rangeslider=dict(visible=True), 
            type="date"
        ),
        hovermode="x unified"
    )
    
    # Y-Axis Settings
    fig.update_yaxes(title_text="Price (IDR)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1, showgrid=False)
    fig.update_yaxes(title_text="MACD", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="RSI", row=3, col=1, secondary_y=True, range=[0, 100], showgrid=False)

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