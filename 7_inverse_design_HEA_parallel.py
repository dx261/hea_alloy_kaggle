"""
Parallel inverse design algorithm for HEA multiple objective optimization with normalization.
"""

import json
import os
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from util.alloys_features import formula_to_features
from util.base_function import get_chemical_formula
from util.descriptor.magpie import get_magpie_features
from util.ml.dnn import RegressionDNN
from util.ml.dnn import train_dnn_by_dataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim


# Function to maximize the design target (for optimization)
def objective_function(target_values, weights):
    """Maximize the function for multi-objective optimization."""
    assert len(weights) == target_values.shape[1], "Weights and target dimensions mismatch."
    f_max = torch.sum(target_values * torch.tensor(weights, device=target_values.device), dim=1)
    return f_max


# Function to normalize compositions
def normalize_compositions(x, mask):
    x = x * mask  # Apply non-zero mask
    row_sums = torch.sum(x, dim=1, keepdim=True)
    return x / row_sums  # Normalize row-wise


# Min-max normalization
def min_max_normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


# Inverse of min-max normalization
def min_max_denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val


if __name__ == '__main__':
    # Hyperparameters
    early_stopping_patient = 15
    num_iterations = 100
    target_weights = [0.5, 0.5]

    # Load models and configuration
    CFS_model = joblib.load("./model/CFS_model.pkl")
    YS_model = joblib.load("./model/YS_model.pkl")
    HV_model = joblib.load("./model/hardness_model.pkl")
    with open('config.json', 'r') as file:
        config = json.load(file)
    CFS_features = config['best_features_zyj_CFS']
    HV_features = config['HV_features']
    YS_features = config['best_features_zyj_YS']

    # Load dataset
    df_train_predict = pd.read_csv("./data/6_training_prediction.csv")
    element_col = list(df_train_predict.columns[1:-2])
    Y_cols = list(df_train_predict.columns[-2:])
    features = element_col
    input_dimension = len(features)
    output_dimension = len(Y_cols)

    # Normalize target values
    y_min = df_train_predict[Y_cols].min()
    y_max = df_train_predict[Y_cols].max()
    df_train_predict[Y_cols] = min_max_normalize(df_train_predict[Y_cols], y_min, y_max)

    # Initialize the DNN model
    model_dnn = RegressionDNN(input_dim=input_dimension, output_dim=output_dimension)
    train_dnn_by_dataset(df_train_predict, model_dnn, features_col=element_col, targets_col=Y_cols)

    # Prepare dataset for optimization
    dataset = df_train_predict.loc[:50, :]
    x_start = dataset[features].values.astype(float)  # Initial input compositions
    x_start_no_zero = (x_start > 0).astype(int)  # Mask for non-zero elements

    # Convert to PyTorch tensors
    x_start = torch.tensor(x_start, dtype=torch.float32, requires_grad=True)
    x_start_no_zero = torch.tensor(x_start_no_zero, dtype=torch.float32)

    # Optimization loop for all samples in parallel
    optimizer = optim.Adam([x_start], lr=0.001)
    best_loss = float('inf')
    counter = 0

    for i in range(num_iterations):
        y = model_dnn(x_start)  # Predict normalized targets
        combined_loss = -torch.mean(objective_function(y, target_weights))  # Negative for maximization

        optimizer.zero_grad()  # Clear gradients
        combined_loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        # Apply constraints
        x_start.data.clamp_(0, 1)  # Ensure values between 0 and 1
        x_start.data = normalize_compositions(x_start.data, x_start_no_zero)  # Normalize compositions

        # Early stopping
        if combined_loss.item() < best_loss:
            best_loss = combined_loss
            counter = 0
        else:
            counter += 1
        if counter >= early_stopping_patient:
            print(f"Early stopping at iteration {i + 1}")
            break

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: Combined Loss = {combined_loss.item()}")

    # Final results
    y_normalized = model_dnn(x_start).detach().numpy()
    y_denormalized = min_max_denormalize(y_normalized, y_min.values, y_max.values)  # Inverse normalization
    df_trace = pd.DataFrame(x_start.detach().numpy(), columns=element_col)
    df_trace[Y_cols] = y_denormalized
    df_trace['objective_function'] = objective_function(torch.tensor(y_denormalized), target_weights).numpy()
    formulas = get_chemical_formula(df_trace[element_col])
    df_trace['formula'] = formulas

    formulas = get_chemical_formula(df_trace[element_col])
    df = pd.DataFrame({"formula": formulas})
    df.to_csv("./data/formula_tmp.csv", index=False)
    # generate magpie features
    df_magpie = get_magpie_features("formula_tmp.csv", data_path="./data/", alloy_features=True)
    alloy_feature = formula_to_features(df['formula'])
    alloy_feature.to_csv(f"./data/df_temp_alloy.csv", index=False)
    X_sampling = pd.concat([df_magpie, alloy_feature], axis=1)
    # add feature calculation here.
    CFS_predict = CFS_model.predict(X_sampling[CFS_features])
    HV_predict = HV_model.predict(X_sampling[HV_features])
    # y3 = YS_model[2].predict(df_trace[element_col])
    df_trace['CFS_ml'] = CFS_predict
    df_trace['HV_ml'] = HV_predict
    # Save results
    df_trace.to_csv("./data/result_parallel.csv", index=False)
    print("Optimization completed. Results saved to './data/result_parallel.csv'.")

    df_train_predict = pd.read_csv("./data/6_training_prediction.csv")
    dataset = df_train_predict.loc[:50, :]
    plt.figure(dpi=300)
    plt.scatter(df_trace['CFS_ml'], df_trace['HV_ml'], c='red', label='opt')
    plt.scatter(dataset['CFS'], dataset['HV'], c='blue', label='train')
    plt.xlabel('CFS')
    plt.ylabel('HV')
    plt.legend()
    plt.show()
