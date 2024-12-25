# Analysis and Prediction of House Prices in Delhi Using Machine Learning
This repository hosts a Jupyter Notebook that provides an in-depth analysis and prediction of house prices in Delhi using various machine learning models. It is a step-by-step guide that includes data exploration, preprocessing, model building, evaluation, and insights derived from the results.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Results](#results)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview
House prices in Delhi are influenced by numerous factors, such as location, property size, amenities, and market trends. This project aims to analyze these factors and predict house prices accurately using machine learning models. The notebook combines data visualization, feature engineering, and predictive analytics to deliver actionable insights and reliable models.

## Key Features
- **Data Analysis**: Explores key factors influencing house prices, identifying trends and correlations within the dataset.
- **Data Preprocessing**: Handles missing values, encodes categorical variables, scales numerical features, and performs feature engineering to improve model performance.
- **Model Building**: Implements several machine learning models, including:
  - Linear Regression
  - Decision Trees
  - Random Forests
  - Gradient Boosting (e.g., XGBoost)
- **Hyperparameter Tuning**: Optimizes model parameters using Grid Search and Cross-Validation to achieve the best performance.
- **Evaluation Metrics**: Assesses model performance using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- **Insights**: Provides an interpretation of the results, highlighting the most impactful features and explaining model predictions.

## Dataset

The dataset used for this project is included in the `data` folder within this repository. It contains detailed information about properties in Delhi, including attributes such as:
- Location
- Property Size
- Number of Bedrooms
- Amenities
- Market Conditions

The data was cleaned and preprocessed to enhance the quality and reliability of the analysis.

## Requirements

To run this project, ensure the following packages are installed:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost

Install dependencies with pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/HarshaEadara/Analysis-and-Prediction-of-House-Prices-in-Delhi-Using-Machine-Learning.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd Analysis-and-Prediction-of-House-Prices-in-Delhi-Using-Machine-Learning
   ```

3. **Access the Dataset**:
   The dataset is available in the `data` folder. Ensure it remains in the same directory for the notebook to function correctly.

4. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook
   ```

5. **Open the Notebook**:
   In the Jupyter interface, open `Analysis_and_Prediction_of_House_Prices_in_Delhi.ipynb`.

6. **Execute the Notebook**:
   Run the cells sequentially to reproduce the analysis and predictions.

## Results

1. **Performance of Models**:
   - Linear Regression showed moderate accuracy but struggled with non-linear relationships in the data.
   - Decision Trees captured complex patterns but were prone to overfitting.
   - Random Forests delivered better generalization, striking a balance between bias and variance.
   - Gradient Boosting models like XGBoost provided the best overall performance with high accuracy and low error rates.

2. **Best Model**:
   - **Gradient Boosting** emerged as the most effective approach, achieving the following metrics:
     - MAE: 50,000 INR
     - RMSE: 70,000 INR
     - R²: 0.85
   - This model's predictions were closest to actual house prices, making it suitable for practical applications.

3. **Feature Importance**:
   - Location and property size were the most significant predictors of house prices.
   - Other influential features included the number of bedrooms and market conditions.

4. **Visual Insights**:
   - The visualizations highlighted trends such as higher prices in premium localities and the impact of property size on pricing.
   - Scatter plots and heatmaps revealed strong correlations between certain features and house prices.

## Contributing

Contributions are welcome! If you'd like to improve the analysis or add new features, feel free to fork the repository and submit a pull request.


