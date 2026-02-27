import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

# --------------------- LOAD TRAINING DATA ---------------------
paths = [
    '/home/uib/predictor/data/weights/3AUV_weights.csv',
    '/home/uib/predictor/data/weights/4AUV_weights.csv',
    '/home/uib/predictor/data/weights/5AUV_weights.csv',
    '/home/uib/predictor/data/weights/6AUV_weights.csv',
]

owa_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
owa_input = owa_df[['auv_count', 'area']].values
owa_output = owa_df[['w1', 'w2', 'w3', 'utility']].values

test_path = "/home/uib/predictor/data/weights/u_max_weights.csv"  
test_df = pd.read_csv(test_path)

# --------------------- SCALING ---------------------
scalers_dict = {}
scaled_inputs = {}

for auv in np.unique(owa_input[:, 0]):
    indices = owa_input[:, 0] == auv
    scaler = StandardScaler()
    scaled_inputs[auv] = scaler.fit_transform(owa_input[indices])
    scalers_dict[auv] = scaler

# --------------------- MODEL DEFINITIONS ---------------------
def create_models():
    return {
        "SVR": MultiOutputRegressor(svm.SVR(kernel='rbf', C=7, epsilon=1.2, gamma=0.1)),
        "Decision Tree": MultiOutputRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=1000)),
        "Polynomial" : MultiOutputRegressor(make_pipeline(PolynomialFeatures(4), LinearRegression())),
        "Lasso": MultiOutputRegressor(Lasso(alpha=0.5)),
    }

models_dict = {auv: create_models() for auv in range(3, 7)}

for auv_count, models in models_dict.items():
    for model in models.values():
        model.fit(scaled_inputs[auv_count], owa_output[owa_input[:, 0] == auv_count])

# --------------------- PREDICTION FUNCTION WITH NORMALIZATION ---------------------
def normalize_weights(weights):
    total = np.sum(weights)
    if total == 0:
        return np.array([10/3, 10/3, 10/3])
    return (weights / total) * 10

def predict_for_testset(test_df, models_dict, scalers_dict):
    rows = []
    for auv in sorted(test_df['auv_count'].unique()):
        test_subset = test_df[test_df['auv_count'] == auv].reset_index(drop=True)
        input_scaled = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
        true_output = test_subset[['w1', 'w2', 'w3', 'utility']].values

        for name, model in models_dict[auv].items():
            preds = model.predict(input_scaled)
            for i in range(len(test_subset)):
                raw_weights = preds[i][:3]
                norm_weights = normalize_weights(raw_weights)
                utility = preds[i][3] if preds.shape[1] > 3 else np.nan

                rows.append({
                    "auv_count": int(auv),
                    "area": test_subset.loc[i, 'area'],
                    "model": name,
                    "true_w1": true_output[i][0],
                    "true_w2": true_output[i][1],
                    "true_w3": true_output[i][2],
                    "true_utility": true_output[i][3],
                    "pred_w1": norm_weights[0],
                    "pred_w2": norm_weights[1],
                    "pred_w3": norm_weights[2],
                    "pred_utility": utility
                })
    return pd.DataFrame(rows)

comparison_df = predict_for_testset(test_df, models_dict, scalers_dict)
comparison_df.to_csv("results/owa_model_predictions_on_test.csv", index=False)
print("\n Predictions on the test dataset saved to 'results/owa_model_predictions_on_test.csv'")

# --------------------- TEST METRICS ---------------------
test_metrics_summary = []

for auv in sorted(test_df['auv_count'].unique()):
    test_subset = test_df[test_df['auv_count'] == auv]
    test_input = scalers_dict[auv].transform(test_subset[['auv_count', 'area']].values)
    test_output = test_subset[['w1', 'w2', 'w3']].values

    for name, model in models_dict[auv].items():
        preds = model.predict(test_input)
        norm_preds = np.apply_along_axis(normalize_weights, 1, preds[:, :3])
        mae = mean_absolute_error(test_output, norm_preds)
        rmse = mean_squared_error(test_output, norm_preds, squared=False)

        test_metrics_summary.append({
            "AUV Count": int(auv),
            "Model": name,
            "MAE": mae,
            "RMSE": rmse
        })

test_metrics_df = pd.DataFrame(test_metrics_summary).sort_values(["AUV Count", "MAE"])
test_metrics_df.to_csv("results/owa_model_test_metrics.csv", index=False)
print("\n Test metrics saved to 'results/regression_models_metrics.csv'")

# --------------------- PLOTS ---------------------
# Add a combined column for AUV and model
test_metrics_df["Group"] = test_metrics_df["AUV Count"].astype(str) + " AUVs - " + test_metrics_df["Model"]

# Sort by AUV Count and MAE
sorted_df = test_metrics_df.sort_values(["AUV Count", "MAE"])
custom_order = sorted_df["Group"].values

# MAE bar plot
plt.figure(figsize=(16, 6))
ax_mae = sns.barplot(x='Group', y='MAE', data=sorted_df, hue='Group', palette='viridis', legend=False)
plt.title("MAE Comparison by Model Within Each AUV Group", fontsize=16)
plt.ylabel("Mean Absolute Error (MAE)", fontsize=14)
plt.xlabel("Model by AUV Group", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add MAE values above bars
for bar in ax_mae.containers[0]:
    height = bar.get_height()
    ax_mae.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# RMSE bar plot
plt.figure(figsize=(16, 6))
ax_rmse = sns.barplot(x='Group', y='RMSE', data=sorted_df, hue='Group', palette='magma', order=custom_order, legend=False)
plt.title("RMSE Comparison by Model Within Each AUV Group", fontsize=16)
plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=14)
plt.xlabel("Model by AUV Group", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add RMSE values above bars
for bar in ax_rmse.containers[0]:
    height = bar.get_height()
    ax_rmse.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
