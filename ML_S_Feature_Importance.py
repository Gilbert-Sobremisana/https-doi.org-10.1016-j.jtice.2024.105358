import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Create a DataFrame with your data, including the new kinetic parameters
data = {
    'Reactant (A)': [9.635, 9.016, 9.729634, 8.36783, 9.635, 9.016, 9.729634, 8.36783],
    #'FReactant (A)': [ 9.902, 7.535,0,0.5820904, 6.843, 6.909, 2.08641962, 0.23077864],
    #'FAloe Emdodin (M2)':[0.463, 1.114, 5.352, 4.105,0.948, 1.577,5.352, 4.105],
    #'FEmodin (M3)': [0.09,0.3,1.451, 1.01, 0.324, 0.424, 1.106, 1.05],
    #'FRhein (M4)': [.956, 3.234,3.72, 3.66, 2.58, 3.508, 3.947, 3.593],
    'Aloe Emdodin (M2)': [0.169, 0.175, 0.908, 1.163, 0.169, 0.175, 0.908, 1.163],
    'Emodin (M3)': [0.073, 0.073, 0.424, 0.425, 0.073, 0.073, 0.424, 0.425],
    'Rhein (M4)': [0.812, 1.172, 3.163, 3.443, 0.812, 1.172, 3.163, 3.443],
    'Chrysophanol (M7)': [0.124, 0.094, 0.429, 0.399, 0.124, 0.094, 0.429, 0.399],
    #'k1 Aloe emodin': [0.003357, 0.0119, 0.01382, 0.01419, 0.008797, 0.0138, 0.025017, 0.01467],
    #'K2 Aloe emodin': [0.01447, 0.008687, 0.003593, 0.004681, 0.007038, 0.009798, 0.002752, 0.004549],
    #'k1 Rhein': [0.001676, 0.01304, 0.005304, 0.002845, 0.01092, 0.01142, 0.00362, 0.00285],
    #'K2 Rhein': [0.011, 0.005084, 0.005729, 0.004296, 0.004913, 0.005939, 0.007449, 0.004296],
    #'k1 Chrysophanol': [0.001772, 0.005867, 0.03756, 0.0859, 0.008496, 0.004814, 0.0085, 0.00866],
    #'K2 Chrysophanol': [0.02021, 0.01619, 0.0003756, 0.01203, 0.007392, 0.01516, 0.003022, 0.007709],
    #'k1 Emodin': [0.0003112, 0.01058, 0.009765, 0.002559, 0.008247, 0.01237, 0.009845, 0.0111],
    #'K2 Emodin': [0.02749, 0.01006, 0.006347, 0.01546, 0.00767, 0.01126, 0.006399, 0.00888],
    'M7_final': [4.389, 5.389, 2.389, 6.389, 8.389, 9.389, 7.389, 3.389]  # You'll need to specify this value
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df.drop(columns=['M7_final'])
y = df['M7_final']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=43)

# Train a Random Forest regressor
rf_model = RandomForestRegressor(n_estimators=50, random_state=43)
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Visualize feature importances
plt.figure(figsize=(10, 6))
colors = ['#d19893', '#b85047', '#a04b4d', '#511e1f', '#794a54']
plt.bar(X.columns, feature_importances, color =colors)
plt.xlabel('$K_{2}$ constants of Compounds')
plt.ylabel('Feature Importance')
plt.title('Feature Importance of K2')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(feature_importances)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate MAE, MSE, RMSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Perform k-fold cross-validation (e.g., 5-fold)
num_folds = 5
cv_scores = cross_val_score(rf_model, X, y, cv=num_folds, scoring='neg_mean_squared_error')
cv_rmse_scores = (-cv_scores) ** 0.5  # Taking the square root of negative MSE to get RMSE

print(f"Cross-Validation RMSE Scores: {cv_rmse_scores}")
print(f"Average RMSE across {num_folds}-fold Cross-Validation: {cv_rmse_scores.mean():.2f}")

# Number of bootstrap iterations
num_iterations = 100

# Initialize an array to store feature importances for each iteration
importance_iterations = np.zeros((num_iterations, len(X.columns)))

# Perform bootstrap resampling
for i in range(num_iterations):
    # Sample with replacement
    bootstrap_sample = df.sample(frac=1, replace=True, random_state=i)

    # Split into features and target
    X_bootstrap = bootstrap_sample.drop(columns=['M7_final'])
    y_bootstrap = bootstrap_sample['M7_final']

    # Train the model
    rf_model_bootstrap = RandomForestRegressor(n_estimators=50, random_state=20)
    rf_model_bootstrap.fit(X_bootstrap, y_bootstrap)

    # Store feature importances
    importance_iterations[i, :] = rf_model_bootstrap.feature_importances_

# Calculate mean and standard deviation of feature importances across iterations
mean_importances = np.mean(importance_iterations, axis=0)
std_importances = np.std(importance_iterations, axis=0)

# Convert to percentage
mean_importances_percentage = mean_importances * 100
std_importances_percentage = std_importances * 100

# Visualize feature importances with error bars
fig = plt.figure(figsize=(10, 6))
patterns = ["*", "+", "X", ".", 'O']

plt.bar(X.columns, mean_importances_percentage, yerr=std_importances_percentage, hatch=patterns, edgecolor='#bd4044', color=colors, capsize=5)
plt.xlabel('Initial Concentration of Compounds', fontname='Times New Roman', fontsize=16)
plt.ylabel('Feature Importance (%)', fontname='Times New Roman', fontsize=16)
plt.title('Feature Importance of K2 with Bootstrap Resampling')
plt.xticks(rotation=45, fontproperties='Times New Roman', fontsize=16)

plt.tight_layout()
plt.show()

# Print mean and standard deviation of feature importances in percentage
print("Mean Feature Importances in Percentage:")
print(mean_importances_percentage)
print("\nStandard Deviation of Feature Importances in Percentage:")
print(std_importances_percentage)