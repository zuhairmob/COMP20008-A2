import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load Movies Dataset
data = pd.read_csv('movies.csv')

# Select relevant features and target variables
features = ['runtime', 'age_certification', 'imdb_votes', 'tmdb_popularity']
target_imdb = 'imdb_score'
target_tmdb = 'tmdb_score'

# Split the data into training and testing sets for IMDb scores
X_imdb = data[features]
y_imdb = data[target_imdb]
X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(
    X_imdb, y_imdb, test_size=0.2, random_state=42)

# Split the data into training and testing sets for TMDB scores
X_tmdb = data[features]
y_tmdb = data[target_tmdb]
X_train_tmdb, X_test_tmdb, y_train_tmdb, y_test_tmdb = train_test_split(
    X_tmdb, y_tmdb, test_size=0.2, random_state=42)

# Standardising features for both models
scaler = StandardScaler()
X_train_imdb = scaler.fit_transform(X_train_imdb)
X_test_imdb = scaler.transform(X_test_imdb)
X_train_tmdb = scaler.fit_transform(X_train_tmdb)
X_test_tmdb = scaler.transform(X_test_tmdb)

# Create linear regression models for IMDb and TMDB scores
model_imdb = LinearRegression()
model_tmdb = LinearRegression()

# Fit the models to the training data for IMDb and TMDB scores
model_imdb.fit(X_train_imdb, y_train_imdb)
model_tmdb.fit(X_train_tmdb, y_train_tmdb)

# Make predictions on the test data for IMDb and TMDB scores
y_pred_imdb = model_imdb.predict(X_test_imdb)
y_pred_tmdb = model_tmdb.predict(X_test_tmdb)

# Evaluate the IMDb model
mse_imdb = mean_squared_error(y_test_imdb, y_pred_imdb)
r2_imdb = r2_score(y_test_imdb, y_pred_imdb)

# Evaluate the TMDB model
mse_tmdb = mean_squared_error(y_test_tmdb, y_pred_tmdb)
r2_tmdb = r2_score(y_test_tmdb, y_pred_tmdb)

# Visualize regression model for IMDb
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test_imdb, y_pred_imdb)
plt.xlabel('Actual IMDb Scores')
plt.ylabel('Predicted IMDb Scores')
plt.title('IMDb Regression Model for movies')
plt.plot([min(y_test_imdb), max(y_test_imdb)], [min(y_test_imdb), max(y_test_imdb)], linestyle='--', color='red', linewidth=2)
plt.grid(True)

# Save the IMDb regression model plot
plt.savefig('imdb_regression_model_movies.png')

# Visualize regression model for TMDB
plt.subplot(1, 2, 2)
plt.scatter(y_test_tmdb, y_pred_tmdb)
plt.xlabel('Actual TMDB Scores')
plt.ylabel('Predicted TMDB Scores')
plt.title('TMDB Regression Mode for moviesl')
plt.plot([min(y_test_tmdb), max(y_test_tmdb)], [min(y_test_tmdb), max(y_test_tmdb)], linestyle='--', color='red', linewidth=2)
plt.grid(True)

# Save the TMDB regression model plot
plt.savefig('tmdb_regression_model_movies.png')

plt.tight_layout()

# Feature Importance Plot for IMDb
feature_importance_imdb = np.abs(model_imdb.coef_)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(features, feature_importance_imdb)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Magnitude')
plt.title('Feature Importance Plot for Movies(IMDB)')
plt.xticks(rotation=45)

# Save the IMDb feature importance plot
plt.savefig('imdb_feature_importance_movies.png')

# Feature Importance Plot for TMDB
feature_importance_tmdb = np.abs(model_tmdb.coef_)

plt.subplot(1, 2, 2)
plt.bar(features, feature_importance_tmdb)
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Magnitude')
plt.title('Feature Importance Plot for Movies(TMDB)')
plt.xticks(rotation=45)

# Save the TMDB feature importance plot
plt.savefig('tmdb_feature_importance_movies.png')

plt.tight_layout()

# Print model performance metrics for IMDb
print("IMDB Model Metrics for Movies:")
print(f"Mean Squared Error (MSE): {mse_imdb:.2f}")
print(f"R-squared (R2): {r2_imdb:.2f}")

# Print model performance metrics for TMDB
print("\nTMDB Model Metrics for Movies:")
print(f"Mean Squared Error (MSE): {mse_tmdb:.2f}")
print(f"R-squared (R2): {r2_tmdb:.2f}")
