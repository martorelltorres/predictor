#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

csv_path = "/home/uib/predictor/data/MLWA_owa_data.csv" #Set your own the path 
output_path = "/home/uib/predictor/results/sensitivity_results_global.csv" #Set your own the path 

df = pd.read_csv(csv_path)

baseline_col = "utility_baseline"
utility_columns = [col for col in df.columns if col.startswith("utility_")]
sensitivity_cols = [col for col in utility_columns if col != baseline_col]

results = []

# Ranking baseline global
df["rank_baseline"] = df[baseline_col].rank(ascending=False, method="min")
baseline_top1 = df.loc[df["rank_baseline"] == 1].index[0]

print("\nTotal configurations:", len(df))

for col in sensitivity_cols:

    # Ranking perturbado global
    df["rank_perturbed"] = df[col].rank(ascending=False, method="min")

    # Spearman entre rankings
    corr, _ = spearmanr(df["rank_baseline"], df["rank_perturbed"])

    perturbed_top1 = df.loc[df["rank_perturbed"] == 1].index[0]
    top1_changes = baseline_top1 != perturbed_top1

    results.append({
        "utility_column": col,
        "spearman_correlation": corr,
        "top1_changes": top1_changes
    })

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print("Global sensitivity analysis completed.")
print(f"Resultados guardados en: {output_path}")

print("\n===== GLOBAL SUMMARY =====")
print(f"Mean Spearman correlation: {results_df['spearman_correlation'].mean():.4f}")
print(f"Top-1 change rate: {results_df['top1_changes'].mean()*100:.2f}%")