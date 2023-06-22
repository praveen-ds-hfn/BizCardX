import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# Data preprocessing
# Handle missing values
data.fillna(data.mean(), inplace=True)

# Remove outliers
Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['Price'] < (Q1 - 1.5 * IQR)) | (data['Price'] > (Q3 + 1.5 * IQR)))]

# Perform feature scaling
scaler = StandardScaler()
data['Scaled_Price'] = scaler.fit_transform(data[['Price']])

# Encode categorical variables
data = pd.get_dummies(data, columns=['Material_Type'])

# Split the data into train and test sets
X = data.drop(['Price', 'Status'], axis=1)
y_reg = data['Price']
y_cls = data['Status']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# Regression model training and evaluation
reg_model = RandomForestRegressor()
reg_model.fit(X_train_reg, y_train_reg)
reg_pred = reg_model.predict(X_test_reg)
reg_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_pred))

# Classification model training and evaluation
cls_model = RandomForestClassifier()
cls_model.fit(X_train_cls, y_train_cls)
cls_pred = cls_model.predict(X_test_cls)
cls_accuracy = accuracy_score(y_test_cls, cls_pred)

# Streamlit web application
st.title("Industrial Copper Modeling")
task = st.selectbox("Select Task", ("Regression", "Classification"))

if task == "Regression":
    st.subheader("Predict Selling Price")
    form = st.form(key='regression-form')
    material_type = form.selectbox('Material Type', data['Material_Type'].unique())
    feature_cols = ['Scaled_Price'] + [col for col in data.columns if 'Material_Type' in col]
    input_data = pd.DataFrame(columns=feature_cols)
    input_data.loc[0] = [0] * len(feature_cols)
    input_data.loc[0, 'Scaled_Price'] = scaler.transform([[st.number_input("Enter Scaled Price")]])[0][0]
    input_data.loc[0, f'Material_Type_{material_type}'] = 1
    predict_button = form.form_submit_button(label='Predict')
    
    if predict_button:
        prediction = reg_model.predict(input_data)[0]
        st.success(f"Predicted Selling Price: {prediction}")

if task == "Classification":
    st.subheader("Lead Classification")
    form = st.form(key='classification-form')
    material_type = form.selectbox('Material Type', data['Material_Type'].unique())
    feature_cols = [col for col in data.columns if col != 'Status'] + [col for col in data.columns if 'Material_Type' in col]
    input_data = pd.DataFrame(columns=feature_cols)
    input_data.loc[
