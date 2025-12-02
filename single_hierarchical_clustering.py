import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt


file_to_analyze = 'B2BSemarangQ1Q2.csv'
month_names = ['Januari', 'Febuari', 'Maret', 'April', 'Mei', 'Juni']

def load_and_clean_header(file_name, month_names):
    df = pd.read_csv(file_name, sep=';', header=None, skipinitialspace=True)
    
    new_cols = {}
    new_cols[1] = 'Sekolah'
    new_cols[3] = 'Produk'
    
    for i, month in enumerate(month_names):
        new_cols[4 + i] = month
    
    df = df.rename(columns=new_cols)
    df_data = df.iloc[2:].copy()
    
    relevant_cols = ['Sekolah', 'Produk'] + month_names
    df_data = df_data[relevant_cols]
    
    df_data = df_data.dropna(subset=['Sekolah'], how='all').reset_index(drop=True)
    
    return df_data

def clean_and_sum(df, cols):
    for col in cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('Rp', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)
    df['Total_Pendapatan'] = df[cols].sum(axis=1)
    return df

df_raw = load_and_clean_header(file_to_analyze, month_names)
df_cleaned = clean_and_sum(df_raw.copy(), month_names)

df_all = df_cleaned.copy() 

df_agg = df_all.groupby(['Sekolah', 'Produk'])['Total_Pendapatan'].sum().reset_index()

idx = df_agg.groupby('Sekolah')['Total_Pendapatan'].idxmax()
df_best_product = df_agg.loc[idx].reset_index(drop=True).rename(columns={'Produk': 'Produk_Terlaris'})

df_pivot = df_agg.pivot_table(index='Sekolah', columns='Produk', values='Total_Pendapatan', fill_value=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pivot)
X_scaled_df = pd.DataFrame(X_scaled, index=df_pivot.index, columns=df_pivot.columns)

linked_matrix = linkage(X_scaled_df, method='ward', metric='euclidean')

plt.figure(figsize=(15, 8))
dendrogram(
    linked_matrix,
    orientation='top',
    labels=df_pivot.index.tolist(),
    distance_sort='descending',
    show_leaf_counts=True
)
plt.title('Dendrogram Hierarchical Clustering Sekolah Berdasarkan Profil Pendapatan Produk (Q1-Q2)')
plt.xlabel('Sekolah')
plt.ylabel('Jarak Euclidean (Dissimilarity)')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('dendrogram_sekolah_produk_Q1Q2.png')
plt.close()

k = 3
clusters = fcluster(linked_matrix, k, criterion='maxclust')

df_pivot['Cluster'] = clusters
df_clustered = df_pivot.reset_index()

df_final = pd.merge(df_clustered[['Sekolah', 'Cluster']], df_best_product[['Sekolah', 'Produk_Terlaris']], on='Sekolah', how='left')

df_final.to_csv('Hasil_Clustering_Sekolah_Q1Q2.csv', index=False)

cluster_profile = df_pivot.groupby('Cluster').mean().round(0)
cluster_profile.to_csv('Profil_Cluster_Pendapatan_Q1Q2.csv')

print("Analisis Q1-Q2 Selesai. Hasil disimpan ke file CSV.")
print("Profil Cluster Berdasarkan Rata-rata Pendapatan Produk (Q1-Q2)")
print(cluster_profile)