import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_interactive_forecast(history_df, pred_base, pred_fuse, dates_fut, emiten):
    """
    Membuat Fan Chart interaktif: Historis + Forecast Baseline vs Fusion.
    """
    last_30 = history_df.tail(60)
    
    fig = go.Figure()

    # 1. Plot Historis
    fig.add_trace(go.Scatter(
        x=last_30['date'], y=last_30['Yt'],
        mode='lines', name='Historical Data',
        line=dict(color='black', width=2)
    ))

    # 2. Plot Baseline
    # Konektor
    fig.add_trace(go.Scatter(
        x=[last_30['date'].iloc[-1], dates_fut[0]],
        y=[last_30['Yt'].iloc[-1], pred_base[0]],
        mode='lines', showlegend=False,
        line=dict(color='#d62728', dash='dot')
    ))
    # Forecast Line
    fig.add_trace(go.Scatter(
        x=dates_fut, y=pred_base,
        mode='lines+markers', name='Baseline Forecast',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8)
    ))

    # 3. Plot Fusion
    # Konektor
    fig.add_trace(go.Scatter(
        x=[last_30['date'].iloc[-1], dates_fut[0]],
        y=[last_30['Yt'].iloc[-1], pred_fuse[0]],
        mode='lines', showlegend=False,
        line=dict(color='#1f77b4', dash='dot')
    ))
    # Forecast Line
    fig.add_trace(go.Scatter(
        x=dates_fut, y=pred_fuse,
        mode='lines+markers', name='Fusion Forecast',
        line=dict(color='#1f77b4', width=3),
        marker=dict(symbol='diamond', size=8)
    ))

    fig.update_layout(
        title=f"Proyeksi Harga Saham: {emiten} (Next 3 Days)",
        xaxis_title="Tanggal",
        yaxis_title="Harga (IDR)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig