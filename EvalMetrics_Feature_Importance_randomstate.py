import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = {
    'A_initial': [9.635, 9.016, 9.729634, 8.36783, 9.635, 9.016, 9.729634, 8.36783],
    'A_final': [ 9.902, 7.535,0,0.5820904, 6.843, 6.909, 2.08641962, 0.23077864],
    'M2_final':[0.463, 1.114, 5.352, 4.105,0.948, 1.577,5.352, 4.105],
    'M3_final': [0.09,0.3,1.451, 1.01, 0.324, 0.424, 1.106, 1.05],
    'M4_final': [.956, 3.234,3.72, 3.66, 2.58, 3.508, 3.947, 3.593],
    'M2_initial': [0.169, 0.175, 0.908, 1.163, 0.169, 0.175, 0.908, 1.163],
    'M3_initial': [0.073, 0.073, 0.424, 0.425, 0.073, 0.073, 0.424, 0.425],
    'M4_initial': [0.812, 1.172, 3.163, 3.443, 0.812, 1.172, 3.163, 3.443],
    'M7_initial': [0.124, 0.094, 0.429, 0.399, 0.124, 0.094, 0.429, 0.399],
    'k1_Aloe_emodin': [0.003357, 0.0119, 0.01382, 0.01419, 0.008797, 0.0138, 0.025017, 0.01467],
    'k2_Aloe_emodin': [0.01447, 0.008687, 0.003593, 0.004681, 0.007038, 0.009798, 0.002752, 0.004549],
    'k1_Rhein': [0.001676, 0.01304, 0.005304, 0.002845, 0.01092, 0.01142, 0.00362, 0.00285],
    'k2_Rhein': [0.011, 0.005084, 0.005729, 0.004296, 0.004913, 0.005939, 0.007449, 0.004296],
    'k1_Chrysophanol': [0.001772, 0.005867, 0.03756, 0.0859, 0.008496, 0.004814, 0.0085, 0.00866],
    'k2_Chrysophanol': [0.02021, 0.01619, 0.0003756, 0.01203, 0.007392, 0.01516, 0.003022, 0.007709],
    'k1_Emodin': [0.0003112, 0.01058, 0.009765, 0.002559, 0.008247, 0.01237, 0.009845, 0.0111],
    'k2_Emodin': [0.02749, 0.01006, 0.006347, 0.01546, 0.00767, 0.01126, 0.006399, 0.00888],
    'M7_final': [4.389, 5.389, 2.389, 6.389, 8.389, 9.389, 7.389, 3.389]  # You'll need to specify this value
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create empty lists to store the evaluation results
random_states = range(1, 50)  # Try random_state values from 1 to 100
mae_scores = []
mse_scores = []
rmse_scores = []
cv_rmse_avg_scores = []

for random_state in random_states:
    # Split the data into features (X) and target (y)
    X = df.drop(columns=['M7_final'])
    y = df['M7_final']

    # Split the data into training and testing sets with the current random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

    # Train a Random Forest regressor
    rf_model = RandomForestRegressor(n_estimators=500, random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Calculate MAE, MSE, RMSE
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Perform k-fold cross-validation (e.g., 5-fold)
    num_folds = 5
    cv_scores = cross_val_score(rf_model, X, y, cv=num_folds, scoring='neg_mean_squared_error')
    cv_rmse_scores = (-cv_scores) ** 0.5  # Taking the square root of negative MSE to get RMSE
    cv_rmse_avg = cv_rmse_scores.mean()

    # Append the results to the lists
    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    cv_rmse_avg_scores.append(cv_rmse_avg)

# Find the index of the lowest MAE
best_mae_index = mae_scores.index(min(mae_scores))
best_mae_random_state = random_states[best_mae_index]

# Find the index of the lowest MSE
best_mse_index = mse_scores.index(min(mse_scores))
best_mse_random_state = random_states[best_mse_index]

# Find the index of the lowest RMSE
best_rmse_index = rmse_scores.index(min(rmse_scores))
best_rmse_random_state = random_states[best_rmse_index]

# Find the index of the lowest average CV RMSE
best_cv_rmse_index = cv_rmse_avg_scores.index(min(cv_rmse_avg_scores))
best_cv_rmse_random_state = random_states[best_cv_rmse_index]

# Plot the evaluation metrics over different random_state values
plt.figure(figsize=(12, 8))
plt.ylim(2.0, 28)
plt.plot(random_states, mae_scores, label='MAE',linewidth=5)
plt.plot(random_states, mse_scores, label='MSE',linewidth=5)
plt.plot(random_states, rmse_scores, label='RMSE',linewidth=5)
plt.plot(random_states, cv_rmse_avg_scores, label='Avg CV RMSE',linewidth=5)
plt.xlabel('Random State',fontsize=18)
plt.ylabel('Error Metrics',fontsize=18)
plt.xticks(fontsize=18)  # Increase font size for x-axis value labels
plt.yticks(fontsize=18)  # Increase font size for y-axis value labels
plt.title('Evaluation Metrics vs. Random State')
plt.legend(fontsize=18, loc='upper left')
plt.grid(True)

# Print the best random_state values for each metric
print(f"Best random_state (based on MAE): {best_mae_random_state}")
print(f"Best random_state (based on MSE): {best_mse_random_state}")
print(f"Best random_state (based on RMSE): {best_rmse_random_state}")
print(f"Best random_state (based on Avg CV RMSE): {best_cv_rmse_random_state}")

# Show the plot
plt.show()
