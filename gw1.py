"""GW1.ipynb
Original file is located at
    https://colab.research.google.com/drive/1zfzZNGjD88kpl_xYZ9Gp90DrpZFOaw0S
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
df = pd.read_csv("/content/kubernetes_resource_allocation_dataset[1].csv")  

# Step 2: Preprocess Data
# Drop irrelevant columns
columns_to_drop = ["pod_name", "namespace", "node_name"]
df = df.drop(columns=columns_to_drop)

df

# Handle missing values (Fill with median for numerical, mode for categorical)
numerical_features = df.select_dtypes(include=['number']).columns
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

df

categorical_features = ["deployment_strategy", "scaling_policy"]
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

df

# Encode 'pod_status'
encoder = LabelEncoder()
df['pod_status_encoded'] = encoder.fit_transform(df['pod_status'])

# Create a mapping of encoded values
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(mapping)

# Convert categorical variables to numeric
encoder = LabelEncoder()
for col in categorical_features + ["pod_status"]:
    df[col] = encoder.fit_transform(df[col])

df



# Define features and target
X = df.drop(columns=["pod_status"])
y = df["pod_status"]

print(y)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 4: Train the Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

pod_status=df["pod_status"].encoder.fit_transform(df["pod_status"])s
print(df['pod_status'],pod_status)

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Manually defined mappings for categorical features
category_mappings = {
    "deployment_strategy": {"Recreate": 0, "RollingUpdate": 1},
    "scaling_policy": {"Manual": 0, "Auto": 1}
}

# Encode pod_status
encoder = LabelEncoder()
df["pod_status_encoded"] = encoder.fit_transform(df["pod_status"])

# Ensure proper mapping
df_mapping = df[["pod_status", "pod_status_encoded"]].drop_duplicates().set_index("pod_status_encoded")

def predict_pod_status(new_data):
    df_new = pd.DataFrame([new_data])

    # Convert categorical features to numerical values
    for col in category_mappings:
        df_new[col] = df_new[col].map(category_mappings[col]).fillna(-1)

    # Ensure feature order matches training data
    df_new = df_new[X.columns]

    # Predict pod status (numeric)
    predicted_num = rf_model.predict(df_new)[0]

    # Map predicted number back to actual status using df_mapping
    predicted_label = df_mapping.loc[predicted_num, "pod_status"] if predicted_num in df_mapping.index else "Unknown"

    return predicted_label

# Example usage:
new_pod = {
    "cpu_request": 1.5, "cpu_limit": 3.6, "memory_request": 3200,
    "memory_limit": 5000, "cpu_usage": 3.3, "memory_usage": 2100,
    "restart_count": 0, "uptime_seconds": 76536,
    "deployment_strategy": "RollingUpdate", "scaling_policy": "Manual",
    "network_bandwidth_usage": 450
}

predicted_status = predict_pod_status(new_pod)
print("Predicted Pod Status:", predicted_status)

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Define the same mapping used during training
status_mapping = {'Pending': 0, 'Running': 1, 'Succeeded': 2, 'Unknown': 3}
reverse_mapping = {v: k for k, v in status_mapping.items()}  # Reverse mapping for decoding predictions

# Define category mappings for categorical features
category_mappings = {
    "deployment_strategy": {"RollingUpdate": 0, "Recreate": 1},
    "scaling_policy": {"Manual": 0, "Auto": 1}
}

def predict_pod_status(new_data, model, X):
    df_new = pd.DataFrame([new_data])

    # Convert categorical variables to numerical values
    for col, mapping in category_mappings.items():
        if col in df_new:
            df_new[col] = df_new[col].map(mapping).fillna(-1)  # Handle unseen categories

    # Ensure feature order matches training data
    df_new = df_new.reindex(columns=X.columns, fill_value=0)

    # Predict pod status (numeric)
    predicted_num = model.predict(df_new)[0]

    # Convert predicted number back to actual pod status using the fixed mapping
    predicted_label = reverse_mapping.get(predicted_num, "Unknown")

    return predicted_label

# Example usage:
new_pod = {
    "cpu_request": 1.5, "cpu_limit": 3.6, "memory_request": 3200,
    "memory_limit": 5000, "cpu_usage": 3.3, "memory_usage": 2100,
    "restart_count": 0, "uptime_seconds": 76536,
    "deployment_strategy": "RollingUpdate", "scaling_policy": "Manual",
    "network_bandwidth_usage": 450
}


predicted_status = predict_pod_status(new_pod, rf_model, X)
print("Predicted Pod Status:", predicted_status)

