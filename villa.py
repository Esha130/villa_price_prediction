import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Preprocessing function
def preprocess_data(result, availability):
    # Convert date columns to datetime format
    result['date'] = pd.to_datetime(result['date'], errors='coerce')
    if 'reservation_date' in result.columns:
        result['reservation_date'] = pd.to_datetime(result['reservation_date'], errors='coerce')
    availability['date'] = pd.to_datetime(availability['date'], errors='coerce')

    # Drop rows with invalid dates
    result = result.dropna(subset=['date'])
    availability = availability.dropna(subset=['date'])

    # Calculate derived features
    result['days_since_checkin'] = (result['date'] - pd.to_datetime('2024-01-01')).dt.days
    if 'reservation_date' in result.columns:
        result['days_since_reservation'] = (result['reservation_date'] - pd.to_datetime('2024-01-01')).dt.days

    # Drop unnecessary columns
    drop_columns = ['net_base_price', 'reservation_date']
    result = result.drop(columns=[col for col in drop_columns if col in result.columns], errors='ignore')

    # Label encode 'villa' and 'SEASONS' columns
    for col in ['villa', 'SEASONS']:
        if col in result.columns:
            le = LabelEncoder()
            result[col] = le.fit_transform(result[col])

    # Ensure only numeric columns are used
    result = result.select_dtypes(include=['number'])

    # Drop rows with missing values in key features
    result = result.dropna()

    return result, availability

# Predict price based on selected date
def predict_price_for_date(date, rf_model, xgb_model, availability_data, data):
    # Filter availability data for the selected date
    selected_date = pd.to_datetime(date).date()
    availability_data['date_only'] = availability_data['date'].dt.date
    available_villas = availability_data[availability_data['date_only'] == selected_date].copy()

    # Remove duplicate villas
    available_villas = available_villas.drop_duplicates(subset=['villa'])

    print(f"\nFiltered available villas for {selected_date}:")
    print(available_villas)

    if available_villas.empty:
        print("No available villas found for the selected date.")
        return pd.DataFrame()

    # Ensure villa columns match in type
    data['villa'] = data['villa'].astype(str)
    available_villas['villa'] = available_villas['villa'].astype(str)

    # Prepare data for prediction
    X = data.drop(columns=['price'])
    predicted_prices_rf = rf_model.predict(X)
    predicted_prices_xgb = xgb_model.predict(X)

    # Create a results dataframe with predictions
    results = pd.DataFrame({
        'villa': data['villa'],  # Include 'villa' for merging
        'predicted_rf_listing_price': predicted_prices_rf,
        'predicted_xgb_listing_price': predicted_prices_xgb,
    })

    # Merge results with available villas
    results = results.merge(available_villas[['villa', 'status']], on='villa', how='inner')

    # Add availability status
    if 'status' in results.columns:
        results['is_available'] = results['status'].apply(lambda x: "Yes" if x == "available" else "No")
    else:
        results['is_available'] = "Unknown"

    # Debugging: Show merged results
    print("\nMerged results with availability:")
    print(results.head())

    # Filter only available villas
    available_villas_results = results[results['is_available'] == "Yes"]

    return available_villas_results

# Train Random Forest and XGBoost models
def train_models(data):
    # Split features and target
    X = data.drop(columns=['price'])
    y = data['price']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Train XGBoost model
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate models
    print("Random Forest Performance:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, rf_predictions)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, rf_predictions)}\n")

    print("XGBoost Performance:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, xgb_predictions)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, xgb_predictions)}")

    return rf_model, xgb_model

# Main Streamlit app
def main():
    import streamlit as st

    st.title("Villa Price Prediction")

    # Load the data
    file = "official_dataset.xlsx"  # Replace with the correct file path
    result = pd.read_excel(file, sheet_name='result')
    availability = pd.read_excel(file, sheet_name='availability')

    # Preprocess data
    data, availability_data = preprocess_data(result, availability)

    # Train models
    rf_model, xgb_model = train_models(data)

    # Date selection
    selected_date = st.date_input("Select a date", min_value=pd.to_datetime('2024-12-01'), max_value=pd.to_datetime('2024-12-31'))
    selected_date_str = pd.to_datetime(selected_date).strftime('%Y-%m-%d')

    # Predict and display prices for the selected date
    predicted_prices = predict_price_for_date(selected_date_str, rf_model, xgb_model, availability_data, data)

    if not predicted_prices.empty:
        st.write(f"Predicted prices for villas on {selected_date_str}:")
        st.dataframe(predicted_prices)
    else:
        st.write(f"No villas are available on {selected_date_str}.")

if __name__ == "__main__":
    main()
