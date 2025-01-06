import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Set the page layout and style
st.set_page_config(page_title="CancerVision Analyzer", layout="wide")

# Add custom CSS to improve visual appearance
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #2c3e50;
        font-size: 3rem;
        font-family: 'Helvetica', sans-serif;
    }
    h2 {
        color: #1abc9c;
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: #ecf0f1;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #16a085;
        color: white;
        font-size: 1.1rem;
        border-radius: 5px;
        height: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function to visualize feature importance
def display_feature_importance(title, feature_importance_df):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#3498db')
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Display title
st.title("CancerVision Analyzer")

# Combined results storage
results = []

# Process Dataset 1
st.header("Dataset 1")
maaz_data = pd.read_excel("Lung_Cancer_Dataset.xlsx")
maaz_data['GENDER'] = maaz_data['GENDER'].map({'M': 1, 'F': 0})
maaz_data['LUNG_CANCER'] = maaz_data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
if 'PEER_PRESSURE' in maaz_data.columns:
    maaz_data = maaz_data.drop(columns=['PEER PRESSURE'])
X1 = maaz_data.drop('LUNG_CANCER', axis=1)
y1 = maaz_data['LUNG_CANCER']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train1, y_train1)
y_pred1 = rf1.predict(X_test1)
accuracy1 = accuracy_score(y_test1, y_pred1)
feature_importances1 = pd.DataFrame({
    'Feature': X1.columns,
    'Importance': rf1.feature_importances_
}).sort_values(by='Importance', ascending=False)
results.append(("Maaz AI", "Random Forest", accuracy1, feature_importances1))

st.write(f"Accuracy: **{accuracy1 * 100:.2f}%**")
st.write("Feature Importances:")
st.dataframe(feature_importances1.style.background_gradient(cmap='coolwarm'))
display_feature_importance('Feature Importance ', feature_importances1)

# Process Dataset 2
st.header("Dataset 2")
raffay_data = pd.read_excel("Breast_Cancer 1.xlsx")
label_encoders = {}
for column in raffay_data.select_dtypes(include=['object']).columns:
    raffay_data[column] = raffay_data[column].astype(str)
    le = LabelEncoder()
    raffay_data[column] = le.fit_transform(raffay_data[column])
    label_encoders[column] = le
selected_features2 = ['AGE', 'RACE', 'MARITAL STATUS']
X2 = raffay_data[selected_features2]
y2 = raffay_data['Status']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
scaler2 = StandardScaler()
X_train2 = scaler2.fit_transform(X_train2)
X_test2 = scaler2.transform(X_test2)
rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train2, y_train2)
feature_importances2 = pd.DataFrame({
    'Feature': selected_features2,
    'Importance': rf2.feature_importances_
}).sort_values(by='Importance', ascending=False)
svc2 = SVC(kernel='linear', random_state=42)
svc2.fit(X_train2, y_train2)
y_pred2 = svc2.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
results.append(("Raffay AI", "SVC", accuracy2, feature_importances2))

st.write(f"Accuracy: **{accuracy2 * 100:.2f}%**")
st.write("Feature Importances:")
st.dataframe(feature_importances2.style.background_gradient(cmap='coolwarm'))
display_feature_importance('Feature Importance', feature_importances2)

# Process Dataset 3
st.header("Dataset 3")
zakriya_data = pd.read_excel("Thyroid_Diff.xlsx")
selected_features3 = ['AGE', 'GENDER', 'SMOKING', 'HX SMOKING', 'HX RADIOTHERAPY']
target_column3 = 'Recurred'
label_encoders = {}
for column in zakriya_data.select_dtypes(include=['object']).columns:
    if column in selected_features3 or column == target_column3:
        le = LabelEncoder()
        zakriya_data[column] = le.fit_transform(zakriya_data[column])
        label_encoders[column] = le
X3 = zakriya_data[selected_features3]
y3 = zakriya_data[target_column3]
scaler3 = StandardScaler()
X3 = scaler3.fit_transform(X3)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
log_reg3 = LogisticRegression(random_state=42, max_iter=500)
log_reg3.fit(X_train3, y_train3)
predictions3 = log_reg3.predict(X_test3)
accuracy3 = accuracy_score(y_test3, predictions3)
feature_importances3 = pd.DataFrame({
    'Feature': selected_features3,
    'Importance': np.abs(log_reg3.coef_[0])
}).sort_values(by='Importance', ascending=False)
results.append(("Zakriya AI", "Logistic Regression", accuracy3, feature_importances3))

st.write(f"Accuracy: **{accuracy3 * 100:.2f}%**")
st.write("Feature Importances:")
st.dataframe(feature_importances3.style.background_gradient(cmap='coolwarm'))
display_feature_importance('Feature Importance', feature_importances3)

# Process Dataset 4
st.header("Dataset 4")
naqvi_data = pd.read_excel("lung_cancer_examples.xlsx")
naqvi_data = naqvi_data.drop(columns=['Name', 'Surname'])
target_column4 = 'Result'
X4 = naqvi_data.drop(columns=[target_column4])
y4 = naqvi_data[target_column4]
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)
dt4 = DecisionTreeClassifier(random_state=42)
dt4.fit(X_train4, y_train4)
predictions4 = dt4.predict(X_test4)
accuracy4 = accuracy_score(y_test4, predictions4)
feature_importances4 = pd.DataFrame({
    'Feature': X4.columns,
    'Importance': dt4.feature_importances_
}).sort_values(by='Importance', ascending=False)
results.append(("Naqvi AI", "Decision Tree", accuracy4, feature_importances4))

st.write(f"Accuracy: **{accuracy4 * 100:.2f}%**")
st.write("Feature Importances:")
st.dataframe(feature_importances4.style.background_gradient(cmap='coolwarm'))
display_feature_importance('Feature Importance', feature_importances4)

# Combined Model
st.header("Combined Model")
all_features_list = list(set(
    list(X1.columns) + selected_features2 + selected_features3 + list(X4.columns)
))
combined_data = pd.DataFrame(columns=all_features_list)

for dataset, target in zip([X1, X2, X3, X4], [y1, y2, y3, y4]):
    temp_data = pd.DataFrame(dataset, columns=dataset.columns if isinstance(dataset, pd.DataFrame) else all_features_list[:dataset.shape[1]])
    temp_data = temp_data.reindex(columns=all_features_list, fill_value=0)
    combined_data = pd.concat([combined_data, temp_data], axis=0)

combined_targets = pd.concat([pd.Series(y1).reset_index(drop=True),
                              pd.Series(y2).reset_index(drop=True),
                              pd.Series(y3).reset_index(drop=True),
                              pd.Series(y4).reset_index(drop=True)], axis=0).reset_index(drop=True)

scaler_combined = StandardScaler()
combined_data = scaler_combined.fit_transform(combined_data)
combined_data = pd.DataFrame(combined_data, columns=all_features_list)

X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    combined_data, combined_targets, test_size=0.2, random_state=42
)
rf_combined = RandomForestClassifier(random_state=42)
rf_combined.fit(X_train_combined, y_train_combined)
y_pred_combined = rf_combined.predict(X_test_combined)
accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
feature_importances_combined = pd.DataFrame({
    'Feature': all_features_list,
    'Importance': rf_combined.feature_importances_
}).sort_values(by='Importance', ascending=False)

st.write(f"Accuracy: **{accuracy_combined * 100:.2f}%**")
st.write("Feature Importances:")
st.dataframe(feature_importances_combined.style.background_gradient(cmap='coolwarm'))
display_feature_importance('Feature Importance - Combined Model', feature_importances_combined)
