import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib


# Load the extracted audio features
X = np.load("features.npy", allow_pickle=True)

# Load the labels
y = np.load("labels.npy", allow_pickle=True)


print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# Convert text labels into numbers
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("Classes:", label_encoder.classes_)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


print("Model training complete!")


# Test the model
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


print()
print("Accuracy:", accuracy)


# Detailed results
print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# Save the trained model
joblib.dump(model, "sentra_model.pkl")

# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")


print()
print("Model saved successfully!")
print("sentra_model.pkl")
print("label_encoder.pkl")