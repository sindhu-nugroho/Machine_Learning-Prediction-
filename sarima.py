import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import os

warnings.filterwarnings("ignore")

FILE_Q1Q2 = "B2BSemarangQ1Q2.csv"
FILE_Q3Q4 = "B2BSemarangQ3Q4.csv"

def load_and_transform_data(file_name, month_map, skiprows, target_header_rows):
    
    encodings = ['utf-8', 'latin-1', 'cp1252']
    df = pd.DataFrame()
    
    print(f"-> Mencoba memuat file: {file_name} dengan skiprows={skiprows}")
    for encoding in encodings:
        try:
            df = pd.read_csv(file_name, skiprows=skiprows, header=target_header_rows, encoding=encoding, sep=';')
            print(f"   Berhasil dimuat sebagai CSV (Encoding: {encoding}, Delimiter: ';')")
            break
        except Exception:
            pass
            
    if df.empty:
        print(f"   ERROR: Gagal memuat file {file_name}. Periksa kembali skiprows dan header.")
        return pd.DataFrame()

    try:
        df.dropna(axis=1, how='all', inplace=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col in df.columns:
                if col[0].startswith('Unnamed'):
                    new_cols.append(col[1])
                elif col[1].startswith('Unnamed'):
                    new_cols.append(col[0])
                else:
                    new_cols.append(f'{col[0]}_{col[1]}')
            df.columns = new_cols
        else:
            df.columns = [str(col) for col in df.columns]

        if len(df.columns) < 4:
            raise Exception("DataFrame hanya memiliki sedikit kolom setelah loading. Header/skiprows salah.")
            
        id_vars = [df.columns[1], df.columns[2], df.columns[3]] 

        month_cols = [col for col in df.columns 
             if (any(month in col for month in month_map.keys()) or col.startswith('2025_'))
             and 'Total' not in col 
             ]
        
        if not month_cols:
            raise Exception("Tidak ada kolom bulan yang ditemukan. Header/skiprows salah.")

        df_long = df.melt(
            id_vars=id_vars,
            value_vars=month_cols,
            var_name='Year_Month_Col',
            value_name='Pendapatan'
        )

        df_long['Month_Name'] = df_long['Year_Month_Col'].apply(lambda x: x.split('_')[-1])
        df_long['Bulan'] = df_long['Month_Name'].map(lambda m: month_map.get(m, m))
        df_long['Date'] = pd.to_datetime(df_long['Bulan'].astype(str) + ' 2025', format='%B %Y', errors='coerce')
        
        df_long.rename(columns={id_vars[0]: 'Client_Name'}, inplace=True)
        df_long = df_long[['Client_Name', 'Date', 'Pendapatan']]
        
        df_long['Pendapatan'] = df_long['Pendapatan'].replace('Libur Ramadhan', 0)
        df_long['Pendapatan'] = df_long['Pendapatan'].astype(str).str.replace('Rp', '').str.replace('.', '', regex=False).str.strip()
        df_long['Pendapatan'] = pd.to_numeric(df_long['Pendapatan'], errors='coerce')
        
        return df_long
    
    except Exception as e:
        print(f"Error saat mentransformasi data dari {file_name}: {e}")
        return pd.DataFrame()

def run_sarima_for_client(client_name, ts_client, p=1, d=0, q=1, P=1, D=0, Q=1, s=12):
    
    safe_client_name = client_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace(".", "")
    
    print(f"\n\n========================================================")
    print(f"  ANALISIS UNTUK KLIEN: {client_name}")
    print(f"========================================================")
    
    plt.figure(figsize=(12, 5))
    plt.plot(ts_client, marker='o', linestyle='-')
    plt.title(f'Tren Pendapatan Bulanan: {client_name}')
    plt.xlabel('Bulan')
    plt.ylabel('Pendapatan (Rp)')
    plt.grid(True)
    plt.savefig(f'1_tren_pendapatan_{safe_client_name}.png')
    plt.close()

    try:
        decomposition = seasonal_decompose(ts_client, model='additive', period=12)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f'Dekomposisi Deret Waktu ({client_name})', y=1.02)
        plt.savefig(f'2_dekomposisi_ts_{safe_client_name}.png')
        plt.close()
    except Exception:
        pass 
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    lags_max = min(11, len(ts_client)//2 - 1) 
    sm.graphics.tsa.plot_acf(ts_client, lags=lags_max, ax=axes[0], title=f'ACF ({client_name})')
    sm.graphics.tsa.plot_pacf(ts_client, lags=lags_max, ax=axes[1], title=f'PACF ({client_name})')
    plt.tight_layout()
    plt.savefig(f'3_acf_pacf_plots_{safe_client_name}.png')
    plt.close()

    print(f"\n(Lihat file 3_acf_pacf_plots_{safe_client_name}.png untuk tuning parameter SARIMA)")
    print(f"Menggunakan parameter default: ({p},{d},{q})x({P},{D},{Q},{s})")


    try:
        model = SARIMAX(
            ts_client,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        results.plot_diagnostics(figsize=(15, 12))
        plt.suptitle(f'Diagnostik Residual SARIMA ({client_name})', y=1.02)
        plt.savefig(f'4_diagnostik_sarima_{safe_client_name}.png')
        plt.close()

    except Exception as e:
        print(f"\n--- GAGAL MELATIH MODEL ---")
        print(f"Error SARIMA untuk {client_name}: {e}. Coba ganti parameter.")
        return 

    n_forecast = 6
    forecast = results.get_forecast(steps=n_forecast)
    forecast_df = forecast.summary_frame(alpha=0.05) 

    last_date = ts_client.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq='MS')[1:]
    forecast_df.index = future_dates

    print("\n--- Hasil Prediksi 6 Bulan (Q1 & Q2 2026) ---")
    print(forecast_df[['mean', 'lower 95%', 'upper 95%']].rename(columns={
        'mean': 'Prediksi Rata-rata',
        'lower 95%': 'Batas Bawah 95% CI',
        'upper 95%': 'Batas Atas 95% CI'
    }))

    plt.figure(figsize=(12, 6))
    plt.plot(ts_client, label='Data Historis (2025)', marker='o', linestyle='-')
    plt.plot(forecast_df.index, forecast_df['mean'], label='Prediksi SARIMA (2026)', color='red', marker='x', linestyle='--')
    plt.fill_between(
        forecast_df.index,
        forecast_df['lower 95%'],
        forecast_df['upper 95%'],
        color='red', alpha=0.1,
        label='95% Confidence Interval'
    )

    plt.title(f'Prediksi Pendapatan {client_name} dengan SARIMA')
    plt.xlabel('Bulan')
    plt.ylabel('Pendapatan (Rp)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'5_hasil_prediksi_sarima_{safe_client_name}.png')
    plt.close()
    
    print(f"\nAnalisis {client_name} selesai. Lihat file PNG yang dihasilkan.")
    
month_map = {
    'Januari': 'January', 'Febuari': 'February', 'Maret': 'March', 
    'April': 'April', 'Mei': 'May', 'Juni': 'June',
    'Juli': 'July', 'Agustus': 'August', 'September': 'September', 
    'Oktober': 'October', 'November': 'November', 'Desember': 'December'
}

print("--- Memuat dan Memproses Data ---")

df_q1q2 = load_and_transform_data(FILE_Q1Q2, month_map, skiprows=1, target_header_rows=[0, 1])
df_q3q4 = load_and_transform_data(FILE_Q3Q4, month_map, skiprows=1, target_header_rows=[0, 1])

if df_q1q2.empty or df_q3q4.empty:
    print("\n[BLOKIR] Proses dihentikan karena salah satu atau kedua file gagal dimuat.")
    print("SOLUSI: Periksa kembali nilai 'skiprows' (coba 0) di MAIN BLOCK.")
    exit()
    
df_all = pd.concat([df_q1q2, df_q3q4], ignore_index=True)

df_all.dropna(subset=['Pendapatan'], inplace=True)
df_all = df_all[df_all['Client_Name'].notna()]
df_all = df_all.dropna(subset=['Date']) 

unique_clients = df_all['Client_Name'].unique()

print(f"\nTotal klien unik yang ditemukan: {len(unique_clients)}")

processed_count = 0
for client in unique_clients:
    ts_client_raw = df_all[df_all['Client_Name'] == client].copy()
    
    ts_client = ts_client_raw.set_index('Date').sort_index()
    ts_client = ts_client['Pendapatan'].resample('MS').sum().fillna(0)
    
    if (ts_client > 0).sum() < 6:
        print(f"SKIP: {client} - Data kurang dari 6 bulan non-nol.")
        continue
    
    run_sarima_for_client(client, ts_client)
    processed_count += 1

print(f"\n========================================================")
print(f"PROSES SELESAI. Total {processed_count} klien telah dianalisis.")
print(f"Cek folder Anda untuk file hasil per klien.")
print(f"========================================================")