import pandas as pd
import os

# Cek path
path = os.path.join('data', 'df_fusion.csv')

print(f"📂 Memeriksa file: {os.path.abspath(path)}")

if os.path.exists(path):
    size = os.path.getsize(path)
    print(f"📊 Ukuran File: {size} bytes")
    
    if size == 0:
        print("❌ ERROR FATAL: File KOSONG (0 KB). Silakan download ulang datanya.")
    else:
        try:
            df = pd.read_csv(path)
            print("✅ SUKSES! Data terbaca.")
            print(f"   Jumlah Baris: {len(df)}")
            print(f"   Kolom: {list(df.columns)}")
            print("\nPreview 5 baris pertama:")
            print(df.head())
        except Exception as e:
            print(f"❌ File ada tapi error dibaca: {e}")
else:
    print("❌ File tidak ditemukan di folder data/.")