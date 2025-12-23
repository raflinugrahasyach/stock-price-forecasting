import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from sklearn.preprocessing import MinMaxScaler

# --- KONSTANTA ---
EMITENS = ['ARTO', 'BBCA', 'BBNI', 'BBRI', 'BBTN', 'BMRI', 'BRIS', 'GOTO']
MODEL_FEATS = ['Yt', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
IDX_QUAL = [7, 8, 9, 10]
IDX_QUANT = [0, 1, 2, 3, 4, 5, 6]

# --- CLASSES UNTUK PATCHING ---
class PatchedDTypePolicy:
    def __init__(self, **kwargs):
        self.name = "float32"
    def get_config(self):
        return {"name": "float32"}

class PatchedInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if 'batch_shape' in kwargs: kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        if 'dtype_policy' in kwargs: kwargs.pop('dtype_policy'); kwargs['dtype'] = 'float32'
        if 'sparse' in kwargs: kwargs['sparse'] = bool(kwargs['sparse'])
        super().__init__(**kwargs)

# --- DATA LOADING & MERGING ---

@st.cache_data
def load_shap_data():
    """
    Load data SHAP summary untuk visualisasi interaktif.
    """
    path = os.path.join('data', 'shap_values_summary.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return pd.DataFrame() # Return empty if not found

@st.cache_data
def load_dataset():
    """
    Load Numerik + Sentimen dengan LEFT JOIN agar data harga tidak hilang.
    """
    path_num = os.path.join('data', 'df_numerik_final.csv')
    path_sen = os.path.join('data', 'df_sentiment_features_daily.csv')
    
    # 1. Load Data Numerik (MASTER DATA)
    if os.path.exists(path_num):
        df_num = pd.read_csv(path_num)
        df_num['date'] = pd.to_datetime(df_num['date'])
        # Bersihkan spasi di nama emiten (PENTING!)
        df_num['relevant_issuer'] = df_num['relevant_issuer'].astype(str).str.strip()
    else:
        st.error(f"❌ File Numerik hilang: {path_num}")
        return pd.DataFrame()

    # 2. Load Data Sentimen
    if os.path.exists(path_sen):
        df_sen = pd.read_csv(path_sen)
        df_sen['date'] = pd.to_datetime(df_sen['date'])
        df_sen['relevant_issuer'] = df_sen['relevant_issuer'].astype(str).str.strip()
    else:
        df_sen = pd.DataFrame()

    # 3. Merge Data (LEFT JOIN)
    # Gunakan 'left' agar semua data harga tetap ada, meski sentimen kosong.
    if not df_sen.empty:
        df_final = pd.merge(df_num, df_sen, on=['date', 'relevant_issuer'], how='left')
        
        # Isi NaN Sentimen dengan 0 (Asumsi Netral/Tidak ada berita)
        cols_sentimen = ['X7', 'X8', 'X9', 'X10'] 
        # Cek nama kolom di df_sen, kadang nama aslinya beda, tapi di merge harusnya aman
        # Jika nama kolom di CSV sentimen Anda X7, X8, dst, maka:
        df_final[cols_sentimen] = df_final[cols_sentimen].fillna(0)
    else:
        df_final = df_num
        for col in ['X7', 'X8', 'X9', 'X10']:
            df_final[col] = 0.0

    # 4. RENAME Columns (Mapping Data Baru -> Model Lama)
    rename_map = {
        'Close': 'Yt', 'Open': 'X1', 'High': 'X2', 'Low': 'X3',
        'Volume': 'X4', 'macd': 'X5', 'rsi': 'X6'
    }
    
    available_cols = df_final.columns
    rename_dict_clean = {k: v for k, v in rename_map.items() if k in available_cols}
    df_final = df_final.rename(columns=rename_dict_clean)

    # 5. FINAL CHECK & SORT
    df_final = df_final.sort_values(['relevant_issuer', 'date']).reset_index(drop=True)
    return df_final

@st.cache_data
def load_evaluation_files():
    dm_path = os.path.join('data', 'tabel_dm_test.csv')
    horizon_path = os.path.join('data', 'df_horizon.xlsx')
    df_dm = pd.read_csv(dm_path) if os.path.exists(dm_path) else None
    try: df_horizon = pd.read_excel(horizon_path) if os.path.exists(horizon_path) else None
    except: df_horizon = None
    return df_dm, df_horizon

def load_prediction_model(emiten, scenario):
    try:
        model_filename = f'model_{scenario}_{emiten}.h5'
        model_path = os.path.join('models', model_filename)
        if not os.path.exists(model_path): return None, None

        custom_objects = {'InputLayer': PatchedInputLayer, 'DTypePolicy': PatchedDTypePolicy}
        model = load_model(model_path, custom_objects=custom_objects)

        # Auto-Fit Scaler
        df_full = load_dataset()
        if df_full.empty: return None, None
        df_e = df_full[df_full['relevant_issuer'] == emiten]
        
        # Validasi kolom lengkap
        missing_cols = [c for c in MODEL_FEATS if c not in df_e.columns]
        if missing_cols:
             # st.error(f"Kolom kurang: {missing_cols}") # Debug only
             return None, None

        data_vals = df_e[MODEL_FEATS].values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_vals) 
        
        return model, scaler

    except Exception as e:
        return None, None

def prepare_input_data(df_emiten, window_size=60):
    if len(df_emiten) < window_size: return None
    return df_emiten[MODEL_FEATS].tail(window_size).values.astype('float32')