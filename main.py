import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt

# 1. Loading Data
df = pd.read_csv('retail_sales_dataset.csv')  
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# 2. Aggregating by week
weekly_sales = df.groupby([pd.Grouper(key='Date', freq='W'), 'Product Category']).agg({'Total Amount': 'sum'}).unstack().fillna(0)
weekly_sales.columns = weekly_sales.columns.get_level_values(1)
weekly_sales = weekly_sales.reset_index()
weekly_sales['Date'] = pd.to_datetime(weekly_sales['Date'], errors='coerce')

# Summing sales
weekly_sales['Total'] = weekly_sales[['Beauty', 'Clothing', 'Electronics']].sum(axis=1)

# 3. Extracting time features
weekly_sales['Month'] = weekly_sales['Date'].dt.month
weekly_sales['Week'] = weekly_sales['Date'].dt.isocalendar().week
weekly_sales['Year'] = weekly_sales['Date'].dt.year

# Adding lag features
for lag in range(1, 4):
    weekly_sales[f'Total_lag_{lag}'] = weekly_sales['Total'].shift(lag)

# Adding rolling average
weekly_sales['Total_rolling_mean_3'] = weekly_sales['Total'].shift(1).rolling(window=3).mean()

# Removing missing values
weekly_sales = weekly_sales.dropna()

# Creating features and target variable
features = ['Beauty', 'Clothing', 'Electronics', 'Month', 'Week', 'Year', 
            'Total_lag_1', 'Total_lag_2', 'Total_lag_3', 'Total_rolling_mean_3']
X = weekly_sales[features]
y = weekly_sales['Total']

# 4. Splitting into train/test (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Hyperparameter tuning with GridSearchCV
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.9]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Save the model
best_model.save_model('xgboost_sales_model.json')
print("Model saved as 'xgboost_sales_model.json'")

# 6. Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² on training set: {cv_scores}")
print(f"Mean cross-validation R²: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# 7. Prediction
y_pred = best_model.predict(X_test)

# 8. Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nModel Accuracy (R²): {r2:.2f}")
print(f"MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

# 10. Visualization
plt.figure(figsize=(12, 6))
plt.plot(weekly_sales['Date'].iloc[train_size:], y_test, label='Actual Sales', marker='o', color='blue')
plt.plot(weekly_sales['Date'].iloc[train_size:], y_pred, label='Predicted Sales', marker='o', color='orange')
plt.title('Comparison of Actual vs Predicted Weekly Sales (XGBoost)', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sales_xgboost_gridsearch_comparison.png', dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
cumulative_actual = np.cumsum(y_test)
cumulative_predicted = np.cumsum(y_pred)
plt.plot(weekly_sales['Date'].iloc[train_size:], cumulative_actual, label='Cumulative Actual Sales', marker='o', color='blue')
plt.plot(weekly_sales['Date'].iloc[train_size:], cumulative_predicted, label='Cumulative Predicted Sales', marker='o', color='orange')
plt.title('Cumulative Sales: Actual vs Predicted (XGBoost)', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Sales ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cumulative_sales_xgboost_gridsearch.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
xgb.plot_importance(best_model, importance_type='gain', max_num_features=10)
plt.title('Feature Importance (XGBoost)', fontsize=14, pad=15)
plt.xlabel('F-Score (Gain)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_xgboost_gridsearch.png', dpi=300)
plt.show()