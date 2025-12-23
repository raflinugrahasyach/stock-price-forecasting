import streamlit as st
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- KONSTANTA ---
EMITENS = ['ARTO', 'BBCA', 'BBNI', 'BBRI', 'BBTN', 'BMRI', 'BRIS', 'GOTO']
# Pastikan nama kolom di CSV sesuai dengan yang Anda punya
FEATS = ['Yt', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'] 
IDX_QUAL = [7, 8, 9, 10]
IDX_QUANT = [0, 1, 2, 3, 4, 5, 6]

# --- FUNGSI LOADERS ---

@st.cache_data
def load_dataset():
    """Load data historis utama."""
    path = os.path.join('data', 'df_fusion.csv') # Prioritas 1
    
    if not os.path.exists(path):
         path = os.path.join('data', 'data_final.csv') # Prioritas 2

    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            st.error(f"Error membaca CSV: {e}")
            return pd.DataFrame()
    else:
        st.error(f"File data tidak ditemukan di folder data/")
        return pd.DataFrame()

@st.cache_data
def load_evaluation_files():
    dm_path = os.path.join('data', 'tabel_dm_test.csv')
    horizon_path = os.path.join('data', 'df_horizon.xlsx')
    
    df_dm = pd.read_csv(dm_path) if os.path.exists(dm_path) else None
    
    try:
        df_horizon = pd.read_excel(horizon_path) if os.path.exists(horizon_path) else None
    except:
        df_horizon = None
        
    return df_dm, df_horizon

def load_prediction_model(emiten, scenario):
    try:
        # 1. LOAD MODEL (Langsung load, karena di Cloud versi TF nya nanti kita samakan)
        model_filename = f'model_{scenario}_{emiten}.h5'
        model_path = os.path.join('models', model_filename)
        
        if not os.path.exists(model_path):
            return None, None

        # Load standard
        model = load_model(model_path)

        # 2. AUTO-FIT SCALER (Tetap pakai ini biar aman dari error pickle)
        df_full = load_dataset()
        if df_full.empty: return None, None
            
        df_e = df_full[df_full['relevant_issuer'] == emiten]
        if df_e.empty: return None, None

        data_vals = df_e[FEATS].values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_vals) 
        
        return model, scaler

    except Exception as e:
        # Print error di logs server nanti jika ada masalah
        print(f"❌ Error Loading {emiten}: {str(e)}")
        return None, None

def prepare_input_data(df_emiten, window_size=60):
    if len(df_emiten) < window_size:
        return None
    return df_emiten[FEATS].tail(window_size).values.astype('float32')