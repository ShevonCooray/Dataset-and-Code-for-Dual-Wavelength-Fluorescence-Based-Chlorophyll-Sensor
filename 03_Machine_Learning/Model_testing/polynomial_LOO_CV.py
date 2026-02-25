import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ==========================
# 1️⃣ Load and Preprocess Data
# ==========================

df = pd.read_csv("chl_2000_3694.csv")

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
# 3️⃣ LOO-CV with RMSE
# ==========================

def perform_loo_cv(X, y, name):
    loo = LeaveOneOut()
    mse_scores = []
    actual_values = []
    predicted_values = []
    training_errors = []

    print(f"\n🔹 {name} - Leave-One-Out Cross-Validation Results")
    print("--------------------------------------------------------------------")
    print(f"{'Sample':<10} {'Actual':<15} {'Predicted':<15} {'MSE':<15} {'RMSE':<15}")
    print("--------------------------------------------------------------------")

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        poly = PolynomialFeatures(degree=2, include_bias=True)
        model = make_pipeline(poly, LinearRegression())
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)[0]
        mse = mean_squared_error([y_test[0]], [y_pred])
        rmse = np.sqrt(mse)
        mse_scores.append(mse)

        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        training_errors.append(train_mse)

        actual_values.append(y_test[0])
        predicted_values.append(y_pred)

        print(f"{i+1:<10} {y_test[0]:<15.5f} {y_pred:<15.5f} {mse:<15.5f} {rmse:<15.5f}")

    # Errors and metrics
    prediction_errors = np.array(predicted_values) - np.array(actual_values)
    rmse_values = np.sqrt(mse_scores)
    overall_rmse = np.sqrt(np.mean(prediction_errors**2))
    std_prediction_error = np.std(prediction_errors)

    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(np.sqrt(mse_scores))    #Important
    std_mse = np.std(mse_scores)
    std_rmse = np.std(rmse_values)
    avg_train_mse = np.mean(training_errors)

    print("--------------------------------------------------------------------")
    print(f"{name} - Mean Validation MSE: {avg_mse:.6f}")
    print(f"{name} - Mean Validation RMSE: {avg_rmse:.6f}")
    print(f"{name} - RMSE (LOO-CV): {overall_rmse:.6f}")
    print(f"{name} - Standard Deviation of Validation MSE: {std_mse:.6f}")
    print(f"{name} - Standard Deviation of Validation RMSE: {std_rmse:.6f}")
    print(f"{name} - Mean Training MSE: {avg_train_mse:.6f}")
    print(f"{name} - Standard Deviation of Prediction Errors (σ_Error): {std_prediction_error:.6f}")

    return (mse_scores, rmse_values, actual_values, predicted_values,
            avg_mse, std_mse, avg_train_mse, std_prediction_error, overall_rmse)

print("\n🚀 Running Leave-One-Out Cross Validation (LOO-CV)...")

# Chl-a
results_chla = perform_loo_cv(X_chla_reduced, y_chla, "Chl-a")
(mse_chla_loo, rmse_chla, actual_chla, predicted_chla,
 avg_mse_chla, std_mse_chla, train_mse_chla, std_err_chla, rmse_overall_chla) = results_chla

# Chl-b
results_chlb = perform_loo_cv(X_chlb_reduced, y_chlb, "Chl-b")
(mse_chlb_loo, rmse_chlb, actual_chlb, predicted_chlb,
 avg_mse_chlb, std_mse_chlb, train_mse_chlb, std_err_chlb, rmse_overall_chlb) = results_chlb

# ==========================
# 4️⃣ Scatter Plot with RMSE Error Bars
# ==========================

plt.figure(figsize=(10, 5))
plt.errorbar(actual_chla, predicted_chla, yerr=rmse_chla, fmt='o', color='blue', alpha=0.7,
             label="Chl-a (with RMSE)", capsize=4, capthick=1.2)
plt.errorbar(actual_chlb, predicted_chlb, yerr=rmse_chlb, fmt='o', color='green', alpha=0.7,
             label="Chl-b (with RMSE)", capsize=4, capthick=1.2)

plt.plot([min(actual_chla + actual_chlb), max(actual_chla + actual_chlb)],
         [min(actual_chla + actual_chlb), max(actual_chla + actual_chlb)],
         linestyle="dashed", color="red", label="Ideal Fit")

plt.xlabel("Actual Concentration (mg/L)")
plt.ylabel("Predicted Concentration (mg/L)")
plt.title("Actual vs. Predicted with RMSE Error Bars")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 5️⃣ MSE per Sample
# ==========================

plt.figure(figsize=(10, 5))

plt.bar(range(len(mse_chla_loo)), mse_chla_loo, color='blue', alpha=0.6, label="Chl-a")
plt.bar(range(len(mse_chlb_loo)), mse_chlb_loo, color='green', alpha=0.6, label="Chl-b")
plt.xlabel("Sample Index")
plt.ylabel("MSE")
plt.title("Sample-wise MSE (LOO-CV)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 6️⃣ Residual Plot
# ==========================

residuals_chla = np.array(actual_chla) - np.array(predicted_chla)
residuals_chlb = np.array(actual_chlb) - np.array(predicted_chlb)

plt.figure(figsize=(10, 5))
plt.scatter(predicted_chla, residuals_chla, color='blue', label="Chl-a Residuals", alpha=0.7)
plt.scatter(predicted_chlb, residuals_chlb, color='green', label="Chl-b Residuals", alpha=0.7)
plt.axhline(0, color='red', linestyle="--", label="Zero Residual")
plt.xlabel("Predicted Concentration (mg/L)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# 7️⃣ Overall R² and RMSE Summary
# ==========================

r2_chla = r2_score(actual_chla, predicted_chla)
r2_chlb = r2_score(actual_chlb, predicted_chlb)

print(f"\n📊 Overall R² for Chl-a: {r2_chla:.6f}")
print(f"📊 Overall R² for Chl-b: {r2_chlb:.6f}")
print(f"📏 RMSE for Chl-a (LOO-CV): {rmse_overall_chla:.6f} mg/L")
print(f"📏 RMSE for Chl-b (LOO-CV): {rmse_overall_chlb:.6f} mg/L")
