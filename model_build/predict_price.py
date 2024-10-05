import joblib
import pandas as pd
import numpy as np

# Load crime data
crime_data = pd.read_csv('Datasets/crimes_by_state.csv')

# Load the trained Random Forest model
model = joblib.load('Models/random_forest_model_for_house_price_prediction.joblib')


# Function to calculate monthly mortgage payment and affordability
def calculate_affordability(price, annual_salary, interest_rate=0.06, years=30):
    # Calculate monthly interest rate
    monthly_interest_rate = interest_rate / 12
    # Total number of monthly payments
    total_payments = years * 12
    # Calculate the monthly mortgage payment using the formula
    monthly_payment = price * (monthly_interest_rate * (1 + monthly_interest_rate) ** total_payments) / (
            (1 + monthly_interest_rate) ** total_payments - 1)
    # Calculate 30% of the user's monthly income (affordability threshold)
    affordable_payment = (annual_salary / 12) * 0.30
    # Return whether the house is affordable or not, and the monthly payment amount
    return monthly_payment <= affordable_payment, monthly_payment


# Predict house price (simple form)
def predict_price(state, bedrooms, baths, salary_per_year=None):
    try:
        # Retrieve crime statistics for the state
        crime_stats = crime_data.loc[crime_data['state'].str.lower() == state.lower()]
        if crime_stats.empty:
            return "State not found in dataset."

        property_crime = crime_stats['property_per_100_000'].values[0]
        violence_crime = crime_stats['violence_per_100_000'].values[0]

        # Create input DataFrame for prediction
        input_data = pd.DataFrame({
            'beds': [bedrooms],
            'baths': [baths],
            'property_per_100_000': [property_crime],
            'violence_per_100_000': [violence_crime],
            'state': [state]
        })

        # Predict using Random Forest
        predicted_price_rf = model.predict(input_data)[0]

        # If no salary is provided, return just the predicted price
        if salary_per_year is None:
            return f'Predicted Price for {bedrooms} bedroom, {baths} bathroom house in {state}: ${predicted_price_rf:,.2f}'

        # If salary is provided, calculate affordability
        is_affordable, monthly_payment = calculate_affordability(predicted_price_rf, salary_per_year)

        # Prepare detailed output
        result = (
            f"Predicted Price for {bedrooms} bedroom, {baths} bathroom house in {state}: ${predicted_price_rf:,.2f}\n"
            f"Yearly Income: ${salary_per_year:,.2f}\n"
            f"Monthly Payment: ${monthly_payment:,.2f}\n"
            f"Affordability Status: {'Affordable' if is_affordable else 'Not Affordable'}")

        return result

    except Exception as e:
        return f"Error in prediction: {e}"


# Main function for user input
def main():
    print("Welcome to the House Price Prediction and Affordability Checker!")

    # Get user inputs for prediction
    state = input("Enter the state: ")
    bedrooms = int(input("Enter the number of bedrooms: "))
    baths = int(input("Enter the number of bathrooms: "))

    # Ask the user if they want to check affordability
    check_affordability = input("Do you want to check affordability (yes/no)? ").strip().lower()

    if check_affordability == 'yes':
        yearly_salary = float(input("Enter your yearly salary (after tax): "))
        # Use the function with salary to check affordability
        result = predict_price(state, bedrooms, baths, salary_per_year=yearly_salary)
    else:
        # Use the function without salary to get only the predicted price
        result = predict_price(state, bedrooms, baths)

    # Output the results
    print("\n" + result)


if __name__ == "__main__":
    main()
