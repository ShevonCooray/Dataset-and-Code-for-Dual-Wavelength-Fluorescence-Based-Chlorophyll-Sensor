import joblib
import numpy as np

# Load trained SVD model
svd_chlb = joblib.load("svd_chlb.pkl")

# Extract SVD transformation matrix (11 components)
svd_matrix = svd_chlb.components_  # Shape: (11, 1694)

# Convert to C-array format for Arduino
print("const float SVD_MATRIX[11][1694] = {")
for row in svd_matrix:
    print("  {" + ", ".join(map(str, row)) + "},")
print("};")


