import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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
# **3️⃣ Train Linear Regression on Full Data**
# ==========================

def evaluate_linear_model(X, y, name):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    residuals = y - y_pred
    sigma_residuals = np.std(residuals)

    # Coefficients
    coefficients = model.coef_
    slope = np.linalg.norm(coefficients)

    print(f"\n{name} - Linear Regression Results:")
    print(f"Training R² Score: {r2:.6f}")
    print(f"Training MSE: {mse:.6f}")
    print(f"σ_residuals (Training): {sigma_residuals:.6f}")
    print(f"Slope Magnitude: {slope:.6f}")
    print(f"Coefficients: {coefficients}")
    print(f"Intercept : {model.intercept_:.10f}")

    return mse, r2, slope, sigma_residuals

# Train and evaluate linear model on full dataset for Chl-a
mse_chla, r2_chla, slope_chla, sigma_chla = evaluate_linear_model(X_chla_reduced, y_chla, "Chl-a")

# Train and evaluate linear model on full dataset for Chl-b
mse_chlb, r2_chlb, slope_chlb, sigma_chlb = evaluate_linear_model(X_chlb_reduced, y_chlb, "Chl-b")

# Variance of target variables
print(f"\nChl-a Concentration Variance: {np.var(y_chla):.6f}")
print(f"Chl-b Concentration Variance: {np.var(y_chlb):.6f}")
