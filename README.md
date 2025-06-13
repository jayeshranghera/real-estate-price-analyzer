# real-estate-price-analyzer

A machine learning project to predict **house prices** based on various features such as income, population, and number of rooms — built using Python, pandas, and scikit-learn.

---

## Project Objective

This project aims to analyze real estate data and build a predictive model that estimates housing prices using linear regression.

---

## Dataset Used

-  File: `USA_Housing.csv`
-  Source: Public housing dataset containing 5000 records with features like:
  - `Avg. Area Income`
  - `Avg. Area House Age`
  - `Avg. Area Number of Rooms`
  - `Avg. Area Number of Bedrooms`
  - `Area Population`
  - `Price` (Target)
  - `Address` (Dropped)

---

##  Technologies & Libraries

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- pickle

---

##  Steps Performed

1. **Data Loading** using `src/data_loader.py`
2. **Exploratory Data Analysis (EDA)**  
   - Distribution plot, scatter plots, correlation heatmap
3. **Data Cleaning**  
   - Removed `Address`, checked for nulls
4. **Model Building**  
   - Trained `LinearRegression` model
5. **Model Evaluation**
   - MAE: `80879.10`
   - RMSE: `100444.06`
   - R² Score: `0.92`
6. **Model Saved** using `pickle`

---

##  Sample Output

```text
Model Evaluation Matrices:
MAE(mean absolute error):80879.10
MSE(mean squared error):10089009300.89
RMSE(root mean squared error):100444.06
R2 score:0.92




About Me:-
Developed by Jayesh Ranghera
Passionate about AI, data science, and impactful products.


