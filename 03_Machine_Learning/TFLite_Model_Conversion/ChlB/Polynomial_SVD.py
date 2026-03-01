import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib
import tensorflow as tf

# ==========================
# **1️⃣ Load & Train Models**
# ==========================

# Load dataset
df = pd.read_csv("../../Dataset/chl_2000_3694.csv")

# Extract Chl-a spectrum data
df_chla = df[df["Chl-a Conc (mg/L)"] > 0]  
X_chla = df_chla.iloc[:, :-2].values  
y_chla = df_chla.iloc[:, -2].values   

# Extract Chl-b spectrum data
df_chlb = df[df["Chl-b Conc (mg/L)"] > 0]  
X_chlb = df_chlb.iloc[:, :-2].values  
y_chlb = df_chlb.iloc[:, -1].values   

# ==========================
# **2️⃣ Apply & Save SVD**
# ==========================

n_components = 11  # Ensure this is the same for training & inference
svd_chla = TruncatedSVD(n_components=n_components, random_state=42)
X_chla_reduced = svd_chla.fit_transform(X_chla)

svd_chlb = TruncatedSVD(n_components=n_components, random_state=42)
X_chlb_reduced = svd_chlb.fit_transform(X_chlb)

print(f"Chl-a: Original shape {X_chla.shape} → Reduced shape {X_chla_reduced.shape}")
print(f"Chl-b: Original shape {X_chlb.shape} → Reduced shape {X_chlb_reduced.shape}")


# Save trained SVD models
joblib.dump(svd_chla, "svd_chla.pkl")
joblib.dump(svd_chlb, "svd_chlb.pkl")
print("SVD models saved!")

# ==========================
# **3️⃣ Apply & Save Polynomial Feature Expansion**
# ==========================
# 2nd-order polynomial
poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)  
  
X_chla_expanded = poly.fit_transform(X_chla_reduced)
X_chlb_expanded = poly.fit_transform(X_chlb_reduced)

print("Expanded Chl-a Feature Count:", X_chla_expanded.shape[1])
print("Expanded Chl-b Feature Count:", X_chlb_expanded.shape[1])

# Save trained Polynomial Transformer
joblib.dump(poly, "poly_transform.pkl")
print("Polynomial transformer saved!")

# ==========================
# **4️⃣ Train Polynomial Regression Models**
# ==========================

poly_model_chla = LinearRegression()
poly_model_chla.fit(X_chla_expanded, y_chla)

poly_model_chlb = LinearRegression()
poly_model_chlb.fit(X_chlb_expanded, y_chlb)

# Save trained regression models
joblib.dump(poly_model_chla, "poly_model_chla.pkl")
joblib.dump(poly_model_chlb, "poly_model_chlb.pkl")
print("Regression models saved!")

