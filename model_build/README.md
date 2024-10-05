
# House Price Prediction Model

This project builds and trains machine learning models to predict house prices based on various factors, including the number of bedrooms, bathrooms, crime statistics, and more. The model also includes an affordability analysis based on household income.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Affordability Prediction](#affordability-prediction)
- [Model Saving and Loading](#model-saving-and-loading)
- [Using predict_price.py](#using-predictpricepy-)
## Project Structure

```
.
├── Datasets
│   ├── housing.csv
│   └── crimes_by_state.csv
├── Models
│   ├── classification_model.joblib
│   └── random_forest_model_for_house_price_prediction.joblib
│   └── linear_regression_model_for_house_price_prediction.joblib
├── house_price_prediction.ipynb
├── predict_price.py 
├── README.md

```

## Important Notes
- The Final Model we are using for the price prediction is the 'random_forest_model_for_house_price_predction.joblib' model.
- The Classification Model is only used in the ipynb file but is also exported to the Models Directory 
- predict_price.py can be used to run the Machine learning models without having to run the Jupyter Notebook. Instructions for this is given below

## Dataset

1. **housing.csv**: Contains house-related data such as price, number of bedrooms, bathrooms, etc.
2. **crimes_by_state.csv**: Contains crime data by state, including property and violent crimes per 100,000 people.

## Prerequisites

Ensure you have the following installed:

- Python 3.9
- `pip` (Python package manager)
  
## Setup

1. **Clone the Repository**:
   ```
   git clone https://github.com/nidulamallikarachchi/House-Price-Prediction-Web-Application/tree/main
   cd ML_Model
   ```

2. **Create a Virtual Environment**:
   To keep the dependencies isolated, it's recommended to create a virtual environment.
   ```
   python -m venv env
   ```

3. **Activate the Virtual Environment**:

   - On Windows:
     ```
     .\env\Scripts\activate
     ```

   - On macOS/Linux:
     ```
     source env/bin/activate
     ```

4. **Install Required Dependencies**:
   Install the necessary libraries by running:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

## Training the Model

1. **Load the Dataset**:
   Inside the Jupyter notebook (`house_price_prediction.ipynb`), the following datasets are loaded:
   ```python
   housing_data = pd.read_csv('Datasets/housing.csv')
   crime_data = pd.read_csv('Datasets/crimes_by_state.csv')
   ```

2. **Preprocessing**:
   - Missing values are handled using the `SimpleImputer`.
   - Numerical features are scaled using `StandardScaler`.
   - Categorical features like `state` are one-hot encoded.
   - Outliers are removed using the IQR (Interquartile Range) method for `price`, `beds`, and `baths`.

3. **Feature Engineering**:
   An `affordability` feature is engineered based on house price, median household income, and other parameters.

4. **Model Training**:
   The project uses two regression models:
   
   - **Random Forest Regressor**: Tuned with `GridSearchCV` for hyperparameter optimization.
   - **Linear Regression**: A simple baseline model.

   Example training code for Random Forest:
   ```python
   rf_grid_search.fit(X_train_reg, y_train_reg)
   ```

## Prediction

To predict house prices, use the `predict_house_price` function. Example usage:

```python
predicted_price = predict_house_price('New York', 2, 1)
print(predicted_price)
```

This will output the predicted prices from both the Random Forest and Linear Regression models.

## Evaluation

Model performance is evaluated using the following metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Example evaluation results for Random Forest:

```python
print(f'Mean Squared Error (MSE): {rf_mse}')
print(f'Root Mean Squared Error (RMSE): {rf_rmse}')
print(f'R² Score: {rf_r2}')
```

## Affordability Prediction

You can also predict whether a house is affordable based on income and house price. Example:

```python
result = can_afford_home('New York', 3, 2, 200000)
print(result)
```

This will return the predicted house price and whether it is affordable for the given income.

## Model Saving and Loading

Trained models are saved using `joblib` for future use. To save a model:

```python
joblib.dump(best_rf_regressor, 'random_forest_model_for_house_price_prediction.joblib')
```

To load a model:

```python
model = joblib.load('random_forest_model_for_house_price_prediction.joblib')
```

## Using predict_price.py 
This python file is configured with the 'random_forest_model_for_house_price_prediction.joblib' model to predict house prices or find the affordability according to users Yearly income. 

To run the file use this on the terminal:
```commandline
python predict_price.py
```

To Change the Model to Linear Regression Model, edit the file to:  
```python
model = joblib.load('Models/linear_regression_model_for_house_price_prediction.joblib')
```

## Conclusion

This project demonstrates a full machine learning pipeline for predicting house prices and determining affordability based on state crime statistics and housing data. The Random Forest model performs slightly better than the Linear Regression model based on the evaluation metrics.

Feel free to explore, retrain the models, or adapt this pipeline to your own datasets!
