import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ==========================
# **1️⃣ Load and Preprocess Data**
# ==========================

df = pd.read_csv("../Dataset/chl_2000_3694.csv")

# Extract Chl-a spectrum data
df_chla = df[df["Chl-a Conc (mg/L)"] > 0]  
X_chla = df_chla.iloc[:, :-2].values  
y_chla = df_chla.iloc[:, -2].values   

# Extract Chl-b spectrum data
df_chlb = df[df["Chl-b Conc (mg/L)"] > 0]  
X_chlb = df_chlb.iloc[:, :-2].values  
y_chlb = df_chlb.iloc[:, -1].values   

# ==========================
# **2️⃣ Apply SVD**
# ==========================

n_components = 11

# Apply SVD to Chl-a spectrum
svd_chla = TruncatedSVD(n_components=n_components, random_state=42)
X_chla_reduced = svd_chla.fit_transform(X_chla)

# Apply SVD to Chl-b spectrum
svd_chlb = TruncatedSVD(n_components=n_components, random_state=42)
X_chlb_reduced = svd_chlb.fit_transform(X_chlb)

print(f"Chl-a: Original shape {X_chla.shape} → Reduced shape {X_chla_reduced.shape}")
print(f"Chl-b: Original shape {X_chlb.shape} → Reduced shape {X_chlb_reduced.shape}")

# ==========================
# **3️⃣ Train Random Forest on Full Data**
# ==========================

def evaluate_rf_model(X, y, name):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    residuals = y - y_pred
    sigma_residuals = np.std(residuals)

    print(f"\n{name} - Random Forest Results:")
    print(f"Training R² Score: {r2:.6f}")
    print(f"Training MSE: {mse:.6f}")
    print(f"σ_residuals (Training): {sigma_residuals:.6f}")

    return mse, r2, sigma_residuals

# Train and evaluate RF model on full dataset for Chl-a
mse_chla_rf, r2_chla_rf, sigma_chla_rf = evaluate_rf_model(X_chla_reduced, y_chla, "Chl-a")

# Train and evaluate RF model on full dataset for Chl-b
mse_chlb_rf, r2_chlb_rf, sigma_chlb_rf = evaluate_rf_model(X_chlb_reduced, y_chlb, "Chl-b")

# Variance of target variables
print(f"\nChl-a Concentration Variance: {np.var(y_chla):.6f}")
print(f"Chl-b Concentration Variance: {np.var(y_chlb):.6f}")
