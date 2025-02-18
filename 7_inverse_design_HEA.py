"""
inverse design algorithm for HEA multiple objective optimization.
"""
import json
import os
import pickle
import joblib
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
    """Maximize the function for multi-objective optimization.

    """
    assert len(weights) == len(target_values)
    f_max = sum([target_values[k] * weights[k] for k in range(len(weights))])
    return f_max


if __name__ == '__main__':
    # Initialize the maximum score for comparison during optimization
    # ========================set hyperparameters of design
    early_stopping_patient = 15  # Early stopping of samping
    num_iterations = 100  # max Number of iterations for samping
    target_weights = [0.5, 0.5]
    # =======================end===============================
    max_score = -float('inf')
    best_loss = float('inf')
    # Load machine learning models
    CFS_model = joblib.load("./model/CFS_model.pkl")
    YS_model = joblib.load("./model/YS_model.pkl")
    HV_model = joblib.load("./model/hardness_model.pkl")
    phase_scaler = joblib.load("./model/phase_scaler.pkl")
    Phase_model = joblib.load("./model/phase_model.pkl")
    with open('config.json', 'r') as file:
        config = json.load(file)
    CFS_features = config['best_features_zyj_CFS']
    HV_features = config['HV_features']
    YS_features = config['best_features_zyj_YS']

    ml_model_path_list = [CFS_model, YS_model, HV_model]
    ml_model_features = [CFS_features, YS_features, HV_features]
    data_path = "./data/"
    Y_col = 'HV'
    # init the model and train from dataset
    df_train_predict = pd.read_csv("./data/6_training_prediction.csv")
    element_col = list(df_train_predict.columns[1:-2])
    print("element_col", element_col)
    Y_cols = list(df_train_predict.columns[-2:])
    print("Y_cols", Y_cols)
    assert len(Y_cols) == len(target_weights), "number of Y_cols must be equal to the number of target weights"
    # Define features and target
    features = element_col
    # DNN architecture input size is the number of elements, and output size is the number of targets
    input_dimension = len(features)
    output_dimension = len(Y_cols)
    print("input_dimension", input_dimension)
    print("output_dimension", output_dimension)
    # Load the primary deep neural network model
    model_dnn = RegressionDNN(input_dim=input_dimension, output_dim=output_dimension)

    # first train by dataset
    print("start first training")
    train_dnn_by_dataset(df_train_predict, model_dnn, features_col=element_col, targets_col=Y_cols)

    # Choose an initial sample from the dataset
    dataset = df_train_predict.sort_values(by=Y_col, ascending=True).reset_index(drop=True)
    data_index = 0  # Index of the initial data point
    x_start = dataset.loc[data_index, features].values.astype(float)  # Initial input composition
    x_start_no_zero = np.array(x_start > 0).astype(int)  # Mask for non-zero elements
    print("x_start", x_start)
    # Convert initial input to PyTorch tensor and enable gradient tracking
    x_start = torch.tensor(x_start, dtype=torch.float32, requires_grad=True)
    x_start_no_zero = torch.tensor(x_start_no_zero, requires_grad=False)

    for k in range(2):
        # Initial forward pass
        y = model_dnn(x_start)
        y_true = dataset[Y_col][data_index]  # True target value

        # Define optimizer
        optimizer = optim.Adam([x_start], lr=0.001)
        # Start optimization loop
        counter = 0
        y_trace = []  # List to store outputs during optimization
        x_start_trace = []  # List to store input compositions during optimization

        for i in range(num_iterations):
            # get DNN predictions of targets.
            y = model_dnn(x_start)
            y_trace.append(y.detach().numpy())
            x_start_trace.append(x_start.tolist())  # Store current input composition
            # Calculate combined loss (negative for maximization)
            combined_loss = -objective_function(y, target_weights)
            print("combined_loss", combined_loss)
            # Backpropagation
            optimizer.zero_grad()  # Clear previous gradients
            combined_loss.backward()  # Compute gradients
            # Check for early stopping
            if combined_loss.item() < best_loss:
                best_loss = combined_loss
                print("best_loss", best_loss)
                counter = 0
            else:
                counter += 1
            if counter >= early_stopping_patient:
                print(f"Early stopping at iteration {i + 1}")
                break
            # Update x based on gradients
            optimizer.step()

            # Apply constraints (e.g., ensure all values are between 0 and 1)
            x_start.data.clamp_(0, 1)
            x_start.data = x_start.data * x_start_no_zero  # Apply non-zero mask
            x_start.data /= torch.sum(x_start.data)  # Normalize composition

            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f'Iteration {i + 1}: Combined Loss = {combined_loss.item()}, x = {x_start} y = {y}')

        # Trace output
        df_trace = pd.DataFrame(x_start_trace, columns=element_col)
        df_trace['y_predict_dnn'] = y_trace
        df_trace['objective_function'] = [objective_function(ys, target_weights)for ys in y_trace]
        # Final validation with ML models
        formulas = get_chemical_formula(df_trace[element_col])
        # print(formulas)
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
        df_trace['CFS'] = CFS_predict
        df_trace['HV'] = HV_predict
        # df_trace['Y3'] = y3
        df_trace.to_csv("./data/result.csv", index=True)
        # Display final predictions
        print(df_trace.head())
        train_dnn_by_dataset(df_trace, model_dnn, features_col=element_col, targets_col=Y_cols)
