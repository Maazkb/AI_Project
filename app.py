import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Replace the URL with the raw link to your CSV file
url = 'https://raw.githubusercontent.com/Maazkb/AI_Project/refs/heads/main/Lung_Cancer_Dataset.csv'
data = pd.read_csv(url)

# Encode categorical variables
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Define predictors and target variable
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Extract feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Step 2: Deploy with Streamlit
st.title('Lung Cancer Risk Factor Analysis')
st.write('This app identifies the top factors contributing to lung cancer based on the dataset.')

# Display feature importance
st.subheader('Feature Importance')
st.bar_chart(feature_importances.set_index('Feature'))

# Display the table of feature importance
st.write(feature_importances)

# Display the accuracy score
st.subheader('Model Accuracy')
st.write(f"The accuracy of the Random Forest model is: **{accuracy:.2f}**")
