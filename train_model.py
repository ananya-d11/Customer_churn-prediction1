import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load your dataset
data = pd.read_csv('../dataset/customer_churn.csv')  # Use correct path

# Step 2: Drop target column and encode features
X = pd.get_dummies(data.drop('Churn', axis=1))
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Assuming 'Yes'/'No'

# Step 3: Save model columns
model_columns = X.columns.tolist()
with open('../model/model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Step 7: Save model and scaler
with open('../model/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('../model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model training complete. Files saved: churn_model.pkl, scaler.pkl, model_columns.pkl")
