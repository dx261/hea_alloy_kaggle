"""
合金逆向设计算法：
"""
import os
import pickle
import numpy as np
import torch
from util.ml.dnn import RegressionDNN
from util.ml.dnn import train_dnn_by_dataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim


def load_model(path):
    """Load a pre-trained PyTorch model from disk."""
    model = torch.load(path)
    return model


# Function to maximize the design target (for optimization)
def objective_function(target_values, weights):
    """Maximize the function for multi-objective optimization."""
    assert len(weights) == len(target_values)
    f_max = sum([target_values[k] * weights[k] for k in range(len(weights))])
    return f_max


if __name__ == '__main__':
    # Initialize the maximum score for comparison during optimization
    # ========================set hyperparameters of design
    early_stopping_patient = 15  # Early stopping of samping
    num_iterations = 100  # max Number of iterations for samping
    target_weights = [0.5, 0.3, 0.2]
    # =======================end===============================
    max_score = -float('inf')
    best_loss = float('inf')
    # Load machine learning models
    ml_model_path_list = ["./model/model_0.pkl", "./model/model_1.pkl", "./model/model_cls.pkl"]
    model_list = []
    for path in ml_model_path_list:
        with open(path, 'rb') as f:
            model_list.append(pickle.load(f))
    data_path = "./data/"
    dataset = pd.read_csv(os.path.join(data_path, "370composition.csv"))
    print("rows", dataset.shape[0])
    print("all col", list(dataset.columns))
    Y_col = 'HV'
    element_col = list(dataset.columns[3:])
    # Define features and target
    features = element_col
    # DNN architecture input size is the number of elements, and output size is the number of targets
    input_dimension = len(features)
    output_dimension = len(ml_model_path_list)
    print("input_dimension", input_dimension)
    print("output_dimension", output_dimension)
    # Load the primary deep neural network model
    model_dnn = RegressionDNN(input_dim=input_dimension, output_dim=output_dimension)
    # init the model and train from dataset
    df_init = pd.read_csv("./data/df_init.csv")
    # first train by dataset
    print("start first training")
    train_dnn_by_dataset(df_init, model_dnn, features_col=element_col, targets_col=["HV", "Y2", "Y3"])

    # Choose an initial sample from the dataset
    dataset = dataset.sort_values(by=Y_col, ascending=True).reset_index(drop=True)
    data_index = 100  # Index of the initial data point
    x_start = dataset.loc[data_index, features].values  # Initial input composition
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
            #(weight_1 * y[0] + weight_2 * y[1] + weight_3 * y[2])
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

        # add feature calculation here.
        y1 = model_list[0].predict(df_trace[element_col])
        y2 = model_list[1].predict(df_trace[element_col])
        y3 = model_list[2].predict(df_trace[element_col])
        df_trace['HV'] = y1
        df_trace['Y2'] = y2
        df_trace['Y3'] = y3
        df_trace.to_csv("./data/result.csv", index=True)
        # Display final predictions
        print(df_trace.head())
        train_dnn_by_dataset(df_trace, model_dnn, features_col=element_col, targets_col=["HV", "Y2", "Y3"])
