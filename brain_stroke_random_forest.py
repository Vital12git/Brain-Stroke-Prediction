import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# 1. Load dataset
file_path = r"C:\Users\manda\OneDrive\Desktop\ML Project\brain_stroke.csv"
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()

features = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]
data = data[features + ["stroke"]].copy()

# 2. Impute missing values
for col in data.columns:
  if data[col].dtype == "object":
    data[col] = data[col].fillna(data[col].mode()[0])
  else:
    data[col] = data[col].fillna(data[col].median())

# 3. Encode categorical features
encoders = {}
for col in [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]:
  encoders[col] = LabelEncoder()
  data[col] = encoders[col].fit_transform(data[col].astype(str))

# 4. Create a clean, representative balanced dataset (1:1 Ratio)
stroke_data = data[data["stroke"] == 1]
no_stroke_data = data[data["stroke"] == 0]

# Sample matching number of non-stroke patients
no_stroke_sampled = resample(
    no_stroke_data,
    replace=False,
    n_samples=len(stroke_data),  # Equalize both classes
    random_state=42,
)

# Combine and shuffle
balanced_data = (
    pd.concat([stroke_data, no_stroke_sampled])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

X = balanced_data[features]
y = balanced_data["stroke"]

# 5. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train tuned Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
)
model.fit(X_train, y_train)

# 7. Evaluate model
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n===== MODEL RESULTS =====")
print(f"Accuracy:  {round(acc * 100, 2)}%")
print(f"Precision: {round(prec * 100, 2)}%")
print(f"Recall:    {round(rec * 100, 2)}%")
print(f"F1-Score:  {round(f1 * 100, 2)}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Interactive Prediction Interface
print("\n===== ENTER PATIENT DETAILS =====")
user_data = {}
for col in features:
  if col in encoders:
    print(f"\nAvailable options for {col}: {list(encoders[col].classes_)}")
    val = input(f"Enter {col}: ").strip()
    if val not in encoders[col].classes_:
      print("Invalid value. Please select one of the shown options.")
      raise SystemExit
    user_data[col] = encoders[col].transform([val])[0]
  else:
    while True:
      try:
        user_data[col] = float(input(f"Enter {col}: "))
        break
      except ValueError:
        print("Please enter a valid number.")

input_df = pd.DataFrame([user_data], columns=features)
pred = model.predict(input_df)[0]
risk_score = model.predict_proba(input_df)[0][1]

print("\n===== PREDICTION RESULT =====")
print(f"Calculated Stroke Risk: {round(risk_score * 100, 2)}%")
if pred == 1:
  print("Prediction: Stroke Risk Detected")
else:
  print("Prediction: No Stroke Risk Detected")
