import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Label Encoding Function
def label_encode_columns(dataframe, columns, label_encoders=None):
    if label_encoders is None:
        label_encoders = {}  # Initialize label encoders dictionary if not passed
    for col in columns:
        le = LabelEncoder()
        # Fit and transform for training data
        dataframe[col] = le.fit_transform(dataframe[col].astype(str))
        label_encoders[col] = le  # Store the encoder
    return dataframe, label_encoders

# Train Models
def train_models(data):
    feature_columns = [col for col in data.columns if col != 'price']
    X = data[feature_columns]
    y = data['price']
    X = X.apply(pd.to_numeric, errors='coerce')  # Ensure numeric

    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)

    sarima_model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model = sarima_model.fit(disp=False)

    return rf_model, xgb_model, sarima_model, feature_columns

# Preprocess Data
def preprocess_data(result, availability):
    # Convert 'check_in' and 'check_out' to datetime, if they are not already
    result['check_in'] = pd.to_datetime(result['check_in'], errors='coerce')
    result['check_out'] = pd.to_datetime(result['check_out'], errors='coerce')

    # Fill missing check-in/check-out values with a placeholder for categorical variables
    result['check_in'] = result['check_in'].fillna('Unknown')
    result['check_out'] = result['check_out'].fillna('Unknown')

    # Handle missing numeric values (bedroom_count, ratings, etc.)
    result['bedroom_count'] = result['bedroom_count'].fillna(result['bedroom_count'].median())
    result['ratings'] = result['ratings'].fillna(result['ratings'].mean())
    result['amount'] = result['amount'].fillna(result['amount'].mean())

    # Handle categorical columns like 'villa' and 'city'
    categorical_columns = ['villa', 'SEASONS', 'city']
    result, label_encoders = label_encode_columns(result, categorical_columns)

    # Return all three values
    return result, availability, label_encoders  

# Modify predict function for unseen labels
def predict_price_for_date(date, rf_model, xgb_model, sarima_model, villa_data, availability_data, feature_columns, label_encoders):
    selected_date = pd.to_datetime(date).date()
    availability_data['date_only'] = availability_data['date'].dt.date
    available_villas = availability_data[availability_data['date_only'] == selected_date]

    if available_villas.empty:
        st.write(f"No available villas found for {selected_date}.")
        return pd.DataFrame()

    # Convert 'villa' and 'city' in available_villas to int using Label Encoding
    # Use transform instead of fit_transform to ensure we use the same encoder learned on training data
    available_villas['villa'] = label_encoders['villa'].transform(available_villas['villa'].astype(str))
    available_villas['city'] = label_encoders['city'].transform(available_villas['city'].astype(str))

    # Prepare input data for predictions
    X = villa_data[feature_columns].copy()
    X = X.apply(pd.to_numeric, errors='coerce')

    # Predict prices using models
    predicted_prices_rf = rf_model.predict(X)
    predicted_prices_xgb = xgb_model.predict(X)
    predicted_prices_sarima = sarima_model.predict(start=len(villa_data), end=len(villa_data), dynamic=False)

    results = pd.DataFrame({
        'villa': villa_data['villa'],
        'predicted_rf_listing_price': predicted_prices_rf,
        'predicted_xgb_listing_price': predicted_prices_xgb,
        'predicted_sarima_listing_price': predicted_prices_sarima
    })

    # Merge with available villas for selected date
    results = results.merge(available_villas[['villa', 'status']], on='villa', how='inner')
    results['is_available'] = results['status'].apply(lambda x: "Yes" if x == "available" else "No")

    # Reverse Label Encoding for villa and city columns to display original names
    results['villa'] = label_encoders['villa'].inverse_transform(results['villa'])
    results['city'] = label_encoders['city'].inverse_transform(results['city'])

    available_villas_results = results[results['is_available'] == "Yes"]
    return available_villas_results

# Streamlit App
def main():
    st.title("Villa Price Prediction")

    # Load data
    file = "official_dataset.xlsx"  # Replace with correct file path
    result = pd.read_excel(file, sheet_name='result')
    availability = pd.read_excel(file, sheet_name='availability')

    # Preprocess data
    villa_data, availability_data, label_encoders = preprocess_data(result, availability)

    # Train models
    rf_model, xgb_model, sarima_model, feature_columns = train_models(villa_data)

    # User input for city, villa name, and date
    selected_city = st.selectbox("Select a city", villa_data['city'].unique())
    selected_villa = st.selectbox("Select a villa", villa_data[villa_data['city'] == selected_city]['villa'].unique())
    selected_date = st.date_input("Select a date", min_value=pd.to_datetime('2024-12-01'), max_value=pd.to_datetime('2024-12-31'))
    selected_date_str = pd.to_datetime(selected_date).strftime('%Y-%m-%d')

    # Predict and display prices
    predicted_prices = predict_price_for_date(
        selected_date_str, rf_model, xgb_model, sarima_model, villa_data, availability_data, feature_columns, label_encoders
    )

    if not predicted_prices.empty:
        st.write(f"Predicted prices for villas on {selected_date_str}:")
        st.dataframe(predicted_prices[['villa', 'predicted_rf_listing_price', 'predicted_xgb_listing_price', 'predicted_sarima_listing_price']])
    else:
        st.write(f"No villas are available on {selected_date_str}.")

if __name__ == "__main__":
    main()
