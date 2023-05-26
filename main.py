import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost


xgb_model = joblib.load('xgb_model.pkl')



st.title('Diamond Price Prediction')


st.write('### Explained of Data')
st.write('**carat** : weight of the diamond (0.2--5.01)')

st.write('**cut** : quality of the cut (Fair, Good, Very Good, Premium, Ideal)')

st.write('**color** : diamond colour, from J (worst) to D (best)')

st.write('**clarity** : a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))')

st.write('**x** : length in mm (0--10.74)')

st.write('**y** : width in mm (0--58.9)')

st.write('**z** : depth in mm (0--31.8)')

st.markdown('')
st.markdown('')

st.markdown('##### Input carat value')
carat = st.slider('Carat', min_value=0.2, max_value=5.0, step=0.1)

st.markdown('##### Input Clarity value')
options_clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
clarity = st.selectbox(
    'Clarity',
    options_clarity
)

st.markdown('##### Input Cut value')
cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
cut = st.selectbox(
    'Cut',
    cut_options
)

st.markdown('##### Input Color value')
options_color = ['E', 'I', 'J', 'H', 'F', 'G', 'D']
color = st.selectbox(
    'Color',
     options_color)

st.markdown('##### Input X value (in mm)')
x = st.slider('X', min_value=0.0, max_value=11.0, step=0.1)

st.markdown('##### Input Y value (in mm)')
y = st.slider('Y', min_value=0.0, max_value=59.0, step=0.1)

st.markdown('##### Input Z value (in mm)')
z = st.slider('Z', min_value=0.0, max_value=32.0, step=0.1)

if st.button("Predict"):

    color_mapping = {'E': 1, 'I': 2, 'J': 3, 'H': 4, 'F': 5, 'G': 6, 'D': 7}
    features_color = color_mapping[color]

    clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
    features_clarity = clarity_mapping[clarity]

    cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
    features_cut = cut_mapping[cut]


    # Prepare the input data as a feature vector
    features = [float(carat), float(features_cut), float(features_color), float(features_clarity), float(x), float(y), float(z)]
    input_data = np.array(features).reshape(1, -1)

    # Make the prediction using the loaded model
    # prediction = rf_model.predict(input_data)
    prediction = xgb_model.predict(input_data)

    # Display the prediction result
    st.success(f"The predicted price is {prediction}")
