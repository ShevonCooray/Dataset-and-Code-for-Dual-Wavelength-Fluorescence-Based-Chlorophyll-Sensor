import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression

# ==========================
# **1️⃣ Load Trained Model & Transformers**
# ==========================
# Load the saved SVD & Polynomial transformers
svd_chla = joblib.load("svd_chlb.pkl")  # SVD Model
poly_transform = joblib.load("poly_transform.pkl")  # Polynomial Feature Transformer
linear_model_chla = joblib.load("poly_model_chlb.pkl")  # Only LinearRegression, NOT a Pipeline

# Extract model parameters
coefficients = linear_model_chla.coef_.reshape(-1, 1)  # Reshape to match (78,1)
intercept = np.array([linear_model_chla.intercept_], dtype=np.float32)  # Ensure (1,)

# Ensure transformations are applied in the correct order
n_features = poly_transform.n_output_features_  # Should be 78

# ==========================
# **2️⃣ Define TensorFlow Model Equivalent to Polynomial Regression**
# ==========================
class PolynomialRegressionModel(tf.Module):
    def __init__(self, coefficients, intercept):
        """Initialize a TensorFlow model with trained polynomial regression coefficients."""
        super().__init__()
        self.coefficients = tf.Variable(coefficients, dtype=tf.float32, trainable=False)
        self.intercept = tf.Variable(intercept, dtype=tf.float32, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, n_features], dtype=tf.float32)])
    def predict(self, x_poly):
        """Apply the trained polynomial regression model."""
        y_pred = tf.matmul(x_poly, self.coefficients) + self.intercept  # Matrix multiplication
        return tf.reshape(y_pred, [-1, 1])  # Ensure output shape is (batch_size, 1)

# Instantiate the TensorFlow model
polynomial_tflite_model = PolynomialRegressionModel(coefficients, intercept)

# ==========================
# **3️⃣ Convert to TensorFlow Lite**
# ==========================
# Convert the TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [polynomial_tflite_model.predict.get_concrete_function()]
)
tflite_model = converter.convert()

# Save the TFLite model
with open("chl_b_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Successfully converted and saved as `chl_b_model.tflite`!")
