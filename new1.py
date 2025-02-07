
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import joblib

# Load the diabetes dataset
@st.cache
def load_data():
    df = pd.read_csv('diabetes.csv')
    return df

df = load_data()

# Display dataset and basic info
st.title('Diabetes Prediction Model')

st.write('### Dataset Overview')
st.dataframe(df.head())

# Show summary statistics
st.write('### Statistical Summary')
st.write(df.describe())

# Visualize the correlation heatmap
st.write('### Correlation Matrix Heatmap')
corr = df.corr().round(2)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
st.pyplot()

# Visualize Age distribution using a KDE plot
st.write('### Age Distribution')
plt.figure(figsize=(8, 6))
sns.kdeplot(df['Age'], fill=True, label="Age")
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.legend()
st.pyplot()

# Visualize Outcome distribution
st.write('### Outcome Distribution')
sns.countplot(x='Outcome', data=df)
st.pyplot()

# Data Preprocessing
X = df.drop(['Outcome'], axis=1).values
y = df['Outcome'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=45, stratify=y)

# K-NN model selection based on number of neighbors
st.write('### K-NN Model Training')
k_neighbors = st.slider('Select number of neighbors for K-NN', min_value=1, max_value=20, value=7, step=1)

knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(X_train, y_train)

# Accuracy of the model
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

st.write(f"### Training Accuracy: {train_accuracy * 100:.2f}%")
st.write(f"### Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
st.write('### Confusion Matrix')
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=0.5, cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
st.pyplot()

# Classification Report
st.write('### Classification Report')
class_report = classification_report(y_test, y_pred)
st.text(class_report)

# ROC-AUC
st.write('### ROC Curve')
y_test_pred_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='K-NN (k={})'.format(k_neighbors))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
st.pyplot()

roc_auc = roc_auc_score(y_test, y_test_pred_prob)
st.write(f"### ROC-AUC: {roc_auc * 100:.2f}%")

# Save the trained model
if st.button('Save Model'):
    joblib.dump(knn, 'knn_diabetes_model.pkl')
    st.write("Model saved successfully!")

# Model Inference
st.write('### Model Prediction')
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=5)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('BloodPressure', min_value=0, max_value=200, value=70)
skin_thickness = st.number_input('SkinThickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=30.0)
diabetes_pedigree = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=21, max_value=100, value=40)

input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

if st.button('Predict'):
    prediction = knn.predict(input_data)
    if prediction[0] == 0:
        st.write("The model predicts: No Diabetes")
    else:
        st.write("The model predicts: Diabetes")