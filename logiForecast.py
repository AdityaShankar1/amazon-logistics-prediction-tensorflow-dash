import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ======================
# 1. Load dataset
# ======================
df = pd.read_csv("C:/Users/shank/Desktop/amazon_delivery.csv")

# --- Basic cleanup ---
df = df.dropna()  # Drop rows with missing values (simple cleanup)

# ======================
# 2. Encode categorical features
# ======================
categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ======================
# 3. Prepare features (X) and label (y)
# ======================
# We'll predict Delivery_Time
feature_cols = [
    "Agent_Age", "Agent_Rating", "Store_Latitude", "Store_Longitude",
    "Drop_Latitude", "Drop_Longitude", "Weather", "Traffic",
    "Vehicle", "Area", "Category"
]
X = df[feature_cols].values
y = df["Delivery_Time"].values

# Scale features for stable training
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 4. Define TensorFlow model
# ======================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Single output for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ======================
# 5. Train model
# ======================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    verbose=1
)

# ======================
# 6. Evaluate model
# ======================
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=== MODEL PERFORMANCE ===")
print(f"Mean Absolute Error on Test Data: {mae:.2f}")

# ======================
# 7. Feature correlation (simple, not true NN importance)
# ======================
corr_matrix = pd.DataFrame(X, columns=feature_cols).corrwith(pd.Series(y))
results_df = pd.DataFrame({
    "Feature": corr_matrix.index,
    "Correlation_with_DeliveryTime": corr_matrix.values
}).sort_values(by="Correlation_with_DeliveryTime", ascending=False)

# ======================
# 8. Print results to console (CLI output)
# ======================
print("\n=== DELIVERY TIME CORRELATIONS ===")
print(results_df.to_string(index=False))

# ======================
# 9. Save results to CSV (newly added)
# ======================
output_file = "delivery_insights.csv"
try:
    results_df.to_csv(output_file, index=False)
    print(f"\n[INFO] Results successfully saved to {output_file}")
except Exception as e:
    print(f"[ERROR] Could not save results to CSV: {e}")