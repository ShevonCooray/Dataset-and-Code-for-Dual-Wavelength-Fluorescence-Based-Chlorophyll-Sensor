import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

# ==========================
# 1️⃣ Load Dataset
# ==========================
df = pd.read_csv("../Dataset/chl_2000_3694.csv")

# Chl-a
df_chla = df[df["Chl-a Conc (mg/L)"] > 0]
X_chla = df_chla.iloc[:, :-2].values
y_chla = df_chla.iloc[:, -2].values

# Chl-b
df_chlb = df[df["Chl-b Conc (mg/L)"] > 0]
X_chlb = df_chlb.iloc[:, :-2].values
y_chlb = df_chlb.iloc[:, -1].values

# ==========================
# 2️⃣ Apply SVD
# ==========================
n_components = 11
svd_chla = TruncatedSVD(n_components=n_components, random_state=42)
X_chla_reduced = svd_chla.fit_transform(X_chla)

svd_chlb = TruncatedSVD(n_components=n_components, random_state=42)
X_chlb_reduced = svd_chlb.fit_transform(X_chlb)

print(f"Chl-a: Original shape {X_chla.shape} → Reduced shape {X_chla_reduced.shape}")
print(f"Chl-b: Original shape {X_chlb.shape} → Reduced shape {X_chlb_reduced.shape}")

# ==========================
# 3️⃣ LOO-CV with Metrics
# ==========================
loo = LeaveOneOut()

mse_chla_loo, mse_chlb_loo = [], []
actual_chla, predicted_chla = [], []
actual_chlb, predicted_chlb = [], []

# Chl-a
for train_idx, test_idx in loo.split(X_chla_reduced):
    X_train, X_test = X_chla_reduced[train_idx], X_chla_reduced[test_idx]
    y_train, y_test = y_chla[train_idx], y_chla[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    mse = mean_squared_error([y_test[0]], [y_pred])
    mse_chla_loo.append(mse)
    actual_chla.append(y_test[0])
    predicted_chla.append(y_pred)

# Chl-b
for train_idx, test_idx in loo.split(X_chlb_reduced):
    X_train, X_test = X_chlb_reduced[train_idx], X_chlb_reduced[test_idx]
    y_train, y_test = y_chlb[train_idx], y_chlb[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    mse = mean_squared_error([y_test[0]], [y_pred])
    mse_chlb_loo.append(mse)
    actual_chlb.append(y_test[0])
    predicted_chlb.append(y_pred)

# ==========================
# 4️⃣ Compute Metrics
# ==========================
# Mean MSE and Standard Deviation of MSE
avg_mse_chla = np.mean(mse_chla_loo)
avg_mse_chlb = np.mean(mse_chlb_loo)
std_mse_chla = np.std(mse_chla_loo)
std_mse_chlb = np.std(mse_chlb_loo)

# Mean MSE and Standard Deviation of RMSE
avg_rmse_chla = np.mean(np.sqrt(mse_chla_loo))
avg_rmse_chlb = np.mean(np.sqrt(mse_chlb_loo))
std_rmse_chla = np.std(np.sqrt(mse_chla_loo))
std_rmse_chlb = np.std(np.sqrt(mse_chlb_loo))

# R² Score
r2_chla = r2_score(actual_chla, predicted_chla)
r2_chlb = r2_score(actual_chlb, predicted_chlb)

# Residuals and σ
residuals_chla = np.array(actual_chla) - np.array(predicted_chla)
residuals_chlb = np.array(actual_chlb) - np.array(predicted_chlb)

std_resid_chla = np.std(residuals_chla)
std_resid_chlb = np.std(residuals_chlb)

# Print summary
print(f"\n✅ Chl-a - Mean Validation MSE: {avg_mse_chla:.6f} | Std Dev of MSE: {std_mse_chla:.6f}")
print(f"✅ Chl-b - Mean Validation MSE: {avg_mse_chlb:.6f} | Std Dev of MSE: {std_mse_chlb:.6f}")

print(f"\n✅ Chl-a - Mean Validation RMSE: {avg_rmse_chla:.6f} | Std Dev of RMSE: {std_rmse_chla:.6f}")
print(f"✅ Chl-b - Mean Validation RMSE: {avg_rmse_chlb:.6f} | Std Dev of RMSE: {std_rmse_chlb:.6f}")

print(f"\n✅ Chl-a - R² Score: {r2_chla:.6f}  | σ_residuals: {std_resid_chla:.6f}")
print(f"✅ Chl-b - R² Score: {r2_chlb:.6f}  | σ_residuals: {std_resid_chlb:.6f}")

# ==========================
# 5️⃣ Plot MSE per Sample
# ==========================
plt.figure(figsize=(10, 5))
plt.bar(range(len(mse_chla_loo)), mse_chla_loo, color='blue', alpha=0.6, label="Chl-a")
plt.bar(range(len(mse_chlb_loo)), mse_chlb_loo, color='green', alpha=0.6, label="Chl-b")
plt.xlabel("Test Sample Index")
plt.ylabel("MSE")
plt.title("Sample-wise MSE (LOO-CV)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 6️⃣ Actual vs Predicted
# ==========================
plt.figure(figsize=(10, 5))
plt.scatter(actual_chla, predicted_chla, color='blue', label="Chl-a", alpha=0.7)
plt.scatter(actual_chlb, predicted_chlb, color='green', label="Chl-b", alpha=0.7)

plt.plot([min(actual_chla + actual_chlb), max(actual_chla + actual_chlb)],
         [min(actual_chla + actual_chlb), max(actual_chla + actual_chlb)],
         linestyle="dashed", color="red", label="Ideal Fit")

plt.xlabel("Actual Concentration (mg/L)")
plt.ylabel("Predicted Concentration (mg/L)")
plt.title("Actual vs. Predicted Concentration")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 7️⃣ Residual Plot
# ==========================
plt.figure(figsize=(10, 5))
plt.scatter(predicted_chla, residuals_chla, color='blue', label="Chl-a Residuals", alpha=0.7)
plt.scatter(predicted_chlb, residuals_chlb, color='green', label="Chl-b Residuals", alpha=0.7)
plt.axhline(0, color='red', linestyle="--", label="Zero Residual Line")

plt.xlabel("Predicted Concentration (mg/L)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.legend()
plt.grid(True)
plt.show()
