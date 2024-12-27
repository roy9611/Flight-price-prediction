import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('flight_prices.csv')

# Preview the dataset
print("Dataset Preview:")
print(data.head())

# Handle missing values if any
data.fillna(method='ffill', inplace=True)

# Feature engineering: Convert date column to useful features
def preprocess_date(df, date_column):
    df['Day'] = pd.to_datetime(df[date_column]).dt.day
    df['Month'] = pd.to_datetime(df[date_column]).dt.month
    df['Year'] = pd.to_datetime(df[date_column]).dt.year
    df['Weekday'] = pd.to_datetime(df[date_column]).dt.weekday
    df.drop(columns=[date_column], inplace=True)  # Drop the original date column
    return df

data = preprocess_date(data, 'Date')  # Assuming the date column is named 'Date'

# Encode categorical columns
categorical_cols = ['Airline', 'Source', 'Destination']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features (X) and target variable (y)
X = data.drop(columns=['Price'])  # Assuming 'Price' is the target column
y = data['Price']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Function to predict flight price for new data
def predict_flight_price(new_data):
    # Ensure new data is in the same format as the training data
    for col in categorical_cols:
        if col in new_data:
            new_data[col] = label_encoders[col].transform([new_data[col]])[0]
    new_data = preprocess_date(new_data, 'Date')  # Convert date column
    return model.predict(pd.DataFrame([new_data]))

# Example prediction
new_flight = {
    'Airline': 'Indigo', 
    'Source': 'Mumbai', 
    'Destination': 'Delhi', 
    'Date': '2024-12-31'
}
print("\nPredicted Flight Price for new flight:")
print(predict_flight_price(new_flight))
