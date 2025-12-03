import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("C:\\raker\\B2BSemarangQ1Q2.csv", sep=";", encoding="latin1")
df.columns = df.columns.str.strip()

bulan_cols = ["Januari", "Februari", "Maret", "April", "Mei", "Juni"]
# bulan_cols = ["Juli", "Agustus", "September", "Oktober", "November", "Desember"]

def clean_rupiah(x):
    if pd.isna(x):
        return 0
    x = str(x).replace("Rp", "").replace(".", "").replace(",", "").strip()
    return int(x) if x.isdigit() else 0

for b in bulan_cols:
    if b in df.columns:
        df[b] = df[b].apply(clean_rupiah)

df["Total_Revenue"] = df[bulan_cols].sum(axis=1)

product_dummies = pd.get_dummies(df["Produk"], prefix="PROD", dtype=int)

X = product_dummies
y = df["Total_Revenue"]

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X, y)

importances = model.feature_importances_
fi = pd.DataFrame({
    "Product": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(fi)

fi_sorted = fi.sort_values("Importance", ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_sorted)))

plt.figure(figsize=(12, 7))
plt.barh(fi_sorted["Product"], fi_sorted["Importance"], color=colors)

for i, value in enumerate(fi_sorted["Importance"]):
    plt.text(value + 0.001, i, f"{value:.3f}", va="center", fontsize=10)

plt.title("Produk Kontributor Revenue (Gradient Boosting)", fontsize=16, pad=15)
plt.xlabel("Importance", fontsize=13)
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
