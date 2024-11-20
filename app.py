import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the saved models and preprocessors
with open('best_model.pkl', 'rb') as model_file:
    regression_model = pickle.load(model_file)
with open('preprocessor.pkl', 'rb') as preprocessor_file:
    regression_preprocessor = pickle.load(preprocessor_file)
with open('cmodel.pkl', 'rb') as file:
    classification_model = pickle.load(file)
with open('cscaler.pkl', 'rb') as f:
    classification_scaler = pickle.load(f)
with open('ct.pkl', 'rb') as f:
    classification_ohe = pickle.load(f)

# Define the possible values for the dropdown menus
status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
product_options = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665', 
                   '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407', 
                   '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662', 
                   '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738', 
                   '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

# Streamlit app
st.set_page_config(page_title="ML Model Prediction", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title('Industrial Copper Modeling')

# Task selection
task = st.selectbox('Select Task', ['PRICE PREDICTION', 'STATUS PREDICTION'], key='task_select')

if task == 'PRICE PREDICTION':
    st.header('TO PREDICT THE SELLING PRICE')
    with st.form("regression_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write('<h3 style="color: #1f77b4;">Fill All The Fields For Prediction</h3>', unsafe_allow_html=True)
            status = st.selectbox("Status", status_options, key='status_reg')
            item_type = st.selectbox("Item Type", item_type_options, key='item_type_reg')
            country = st.selectbox("Country", sorted(country_options), key='country_reg')
            application = st.selectbox("Application", sorted(application_options), key='application_reg')
            product_ref = st.selectbox("Product Reference", product_options, key='product_ref_reg')
        with col3:
            st.write(f'<h5 style="color:rgb(0, 153, 153,0.6);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity In Tons (Min: 611112 & Max: 1722207579)", "")
            thickness = st.text_input("Enter Thickness (Min: 0.18 & Max: 400)", "")
            width = st.text_input("Enter Width (Min: 1, Max: 2990)", "")
            customer = st.text_input("Customer ID (Min: 12458, Max: 30408185)", "")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        
        if submit_button:
            try:
                # Convert inputs to float
                input_data = pd.DataFrame([[float(quantity_tons), status, item_type, float(application), float(thickness), float(width), float(country), float(customer), product_ref]],
                                          columns=['quantity tons_log', 'status', 'item type', 'application', 'thickness_log', 'width', 'country', 'customer', 'product_ref'])
                # Preprocess
                input_data_transformed = regression_preprocessor.transform(input_data)
                # Predict
                prediction = regression_model.predict(input_data_transformed)
                st.write('<h3 style="color: #2ca02c;">Predicted Selling Price:</h3>', unsafe_allow_html=True)
                st.write(f'<h4 style="color: #ff7f0e;">{np.exp(prediction[0]):.2f}</h4>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in prediction: {e}")

elif task == 'STATUS PREDICTION':
    st.header('TO PREDICT THE STATUS')
    with st.form("classification_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write('<h3 style="color: #1f77b4;">Fill All The Fields For Prediction</h3>', unsafe_allow_html=True)
            status = st.selectbox("Status", status_options, key='status_clf')
            item_type = st.selectbox("Item Type", item_type_options, key='item_type_clf')
            country = st.selectbox("Country", sorted(country_options), key='country_clf')
            application = st.selectbox("Application", sorted(application_options), key='application_clf')
            product_ref = st.selectbox("Product Reference", product_options, key='product_ref_clf')
        with col3:
            st.write(f'<h5 style="color:rgb(0, 153, 153,0.6);">NOTE: Min & Max given for reference, you can enter any value</h5>', unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity In Tons (Min: 611112 & Max: 1722207579)", "")
            selling_price_log = st.text_input("Enter Selling Price (Min: 0, Max: 500)", "")
            thickness = st.text_input("Enter Thickness (Min: 0.18 & Max: 400)", "")
            width = st.text_input("Enter Width (Min: 1, Max: 2990)", "")
            customer = st.text_input("Customer ID (Min: 12458, Max: 30408185)", "")
            submit_button = st.form_submit_button(label="PREDICT STATUS")
        
        if submit_button:
            try:
                # Encode 'status'
                status_encoded = classification_ohe.transform([[status]]).toarray()
                input_data = np.array([[float(quantity_tons), float(selling_price_log), float(application), float(thickness), float(width), float(country), float(customer), product_ref]])
                input_data_combined = np.concatenate((input_data, status_encoded), axis=1)
                # Scale
                input_data_scaled = classification_scaler.transform(input_data_combined)
                # Predict
                prediction = classification_model.predict(input_data_scaled)
                status_pred = 'Won' if prediction[0] == 1 else 'Lost'
                st.write('<h3 style="color: #2ca02c;">Predicted Status:</h3>', unsafe_allow_html=True)
                st.write(f'<h4 style="color: #ff7f0e;">{status_pred}</h4>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error in prediction: {e}")

# Add task bar at the bottom with name
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<h6 style="text-align: center; color: #808080;">ITS DONE BY FAIZAL</h6>', unsafe_allow_html=True)
