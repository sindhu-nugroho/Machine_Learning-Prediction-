import pandas as pd
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\raker\\B2BSemarangQ1Q2.csv", sep=";", encoding="latin1")
df.columns = df.columns.str.strip()

bulan_cols = ["Januari", "Februari", "Maret", "April", "Mei", "Juni"]

def clean_rupiah(x):
    if pd.isna(x):
        return 0
    x = str(x).replace("Rp", "").replace(".", "").replace(",", "").strip()
    return int(x) if x.isdigit() else 0

for b in bulan_cols:
    df[b] = df[b].apply(clean_rupiah)

df["Total_Revenue"] = df[bulan_cols].sum(axis=1)

product_dummies = pd.get_dummies(df["Produk"], prefix="PROD", dtype=int)

X = product_dummies
y = df["Total_Revenue"]

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

importances = model.feature_importances_
fi = pd.DataFrame({
    "Product": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(fi)

plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10)
plt.show()

