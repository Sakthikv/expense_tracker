# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset (replace with actual disease surveillance data)
data = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Step 2: Data Preprocessing
# Check for missing values
print("\nChecking for missing values before any cleaning:")
print(data.isnull().sum())

# Step 3: Handle missing values in 'Outcome Variable'
# Drop rows with NaN in 'Outcome Variable'
data = data.dropna(subset=['Outcome Variable'])
print("\nDataset shape after dropping rows with NaNs in 'Outcome Variable':", data.shape)

# Check for missing values after dropping NaNs
print("\nMissing values after dropping NaNs in 'Outcome Variable':")
print(data.isnull().sum())

# Step 4: Map 'positive' and 'negative' to 1 and 0 in 'Outcome Variable'
print("\nUnique values in 'Outcome Variable' before mapping:")
print(data['Outcome Variable'].unique())

# Convert 'Outcome Variable' to lowercase for consistent mapping, then map values
data['Outcome Variable'] = data['Outcome Variable'].str.lower().map({'positive': 1, 'negative': 0})

print("\nUnique values in 'Outcome Variable' after mapping:")
print(data['Outcome Variable'].unique())

# Step 5: Handling categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical columns found:", categorical_cols)

# Drop any remaining categorical columns for simplicity
data = data.drop(columns=categorical_cols)

# Check dataset after encoding
print("\nDataset shape after encoding:", data.shape)

# Step 6: Splitting data into X and y
X = data.drop('Outcome Variable', axis=1)  # Features
y = data['Outcome Variable']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Initialize and train SVM model
svm_model = SVC(kernel='rbf')  # Radial Basis Function (RBF) kernel
svm_model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = svm_model.predict(X_test)

# Step 10: Evaluate the model
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Step 11: Visualizing the SVM Results
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix for SVM Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
