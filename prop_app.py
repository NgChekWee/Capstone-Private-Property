# Import Libraries
import streamlit as st
import pandas as pd

# Set Page configuration
st.set_page_config(page_title='Predict Private Property Prices in CCR', layout='wide', initial_sidebar_state='expanded')

# Set title of the app
st.title('Predict Private Property Prices in the CCR')

st.write('This predictive model is built using multi-factors linear regression model.')
st.write('It is based on a 5-years transaction dataset from 2018 to 2022.')
st.write('It has a R-square value of 0.55 and has a RMSE of around 11.5% of the prices mean.')

# Load data
prices = pd.read_csv('app_prices.csv')
rental = pd.read_csv('app_rental.csv')
prices["Sale_Date"]=pd.to_datetime(prices["Sale_Date"])
rental["Date"]=pd.to_datetime(rental["Date"])

# Set input widgets
st.sidebar.subheader('Select property attributes')
area = st.sidebar.number_input('Area (sqft)', 400, 5000)
property_type = st.sidebar.slider('Property Type (Apartment = 1 ; Condominium = 2)', 1, 2)
tenure = st.sidebar.slider('Tenure (99-leasehold = 1 ; Freehold = 2)', 1, 2)
unit_age = st.sidebar.slider('Age of unit (years)', 0, 10, 50)
dist_to_mrt = st.sidebar.number_input('Distance to MRT (meters)', 0, 3000)
listing_price = st.sidebar.number_input('Unit Listing Price (psf)', 0, 6000)

# Generate prediction based on user selected attributes
y_pred = round(((0.2964 * area) - (89.3533 * property_type) + (384.7917 * tenure) - (74.8624 * unit_age) - (0.0396 * dist_to_mrt) + 2061.99),2)
valuation = round(((listing_price-y_pred)/y_pred *100),2)

# Display EDA
st.subheader('Exploratory Data Analysis')
st.write('Mean resale prices, transaction volumes and mean rental in psf of private properties in CCR from 2018 to 2022')

prices_grp = prices.groupby(by="Sale_Date")["Unit_Price"].mean()
vol_grp = prices.groupby(by="Sale_Date")["Unit_Price"].count()
rental_grp=rental.groupby(by="Date")["Rental psf"].mean()

st.line_chart(prices_grp)
st.bar_chart(vol_grp)
st.line_chart(rental_grp)

# Print input features
st.subheader('Input Parameters for Prediction')
input_feature = pd.DataFrame([[area, property_type, tenure, unit_age, dist_to_mrt]],
                            columns=['Area (sqft)', 'Property Type', 'Tenure', 'Age of unit (years)', 'Distance to MRT (m)'])
st.write(input_feature)

# Print predicted flower species
st.subheader('Prediction')
st.metric(label='Predicted property price in psf is :', value= y_pred, delta=valuation)