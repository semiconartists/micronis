import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell
def _():
    # File Reading Packages 
    import marimo as mo
    import pandas as pd
    import glob
    import os
    import numpy as np
    from scipy.stats import linregress
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import LabelEncoder
    import torch.nn as nn
    import torch.optim as optim
    import plotly.express as px
    return (
        DataLoader,
        LabelEncoder,
        TensorDataset,
        glob,
        linregress,
        mo,
        nn,
        np,
        optim,
        os,
        pd,
        torch,
        train_test_split,
    )


@app.cell
def _(glob, os, pd):
    def load_concat(directory, pattern):
        search_pattern = os.path.join(directory, pattern)
        parquet_files = glob.glob(search_pattern)
        ls_dataframes = []
        for f_path in parquet_files:
            try:
                temp_df = pd.read_parquet(f_path)
                ls_dataframes.append(temp_df)
            except Exception as e:
                pass
        if not ls_dataframes:
            return pd.DataFrame()
        combined_run_df = pd.concat(ls_dataframes, ignore_index=True)
        return combined_run_df



    return (load_concat,)


@app.cell
def _(load_concat):
    all_run_data_df = load_concat("./data/train/", "run_data_*.parquet")
    if not all_run_data_df.empty:
        all_run_data_df.info()
    all_run_data_df.shape
    return (all_run_data_df,)


@app.cell
def _(all_run_data_df):
    # Number of unique values for each var
    num_unique_runs = all_run_data_df.nunique()
    num_unique_runs

    '''
    Since number of run ids are < 5000 and sensor name is < 50, it is reasonable to pivot the table. 
    '''

    return


@app.cell
def _(all_run_data_df):
    # Pivot the Data 
    all_run_data_df_sorted = all_run_data_df.sort_values(by = ["Run ID", "Time Stamp"])
    # Pivoting Sensor Data
    try:
        pivoted_df = all_run_data_df_sorted.pivot_table(
            index = ['Run ID', 'Time Stamp'],
            columns = 'Sensor Name', 
            values = 'Sensor Value'
        )
    except Exception as e:
        pass

    pivoted_df.head()

    # Handle NA values
    def fill_na(group):
        return group.ffill().bfill()
    if not pivoted_df.empty:
        pivot_df_filled = pivoted_df.groupby(level = 'Run ID', group_keys=False).apply(fill_na)
        rem_NAs = pivot_df_filled.isnull().sum().sum()
    rem_NAs
    pivot_df_filled = pivot_df_filled.reset_index()
    pivot_df_filled.head()
    return (pivot_df_filled,)


@app.cell
def _(load_concat):
    incoming_run_data_df =  load_concat("./data/train/", "incoming_run_data_*.parquet")
    if not incoming_run_data_df.empty:
        incoming_run_data_df.info()
    incoming_run_data_df.nunique()
    return (incoming_run_data_df,)


@app.cell
def _(incoming_run_data_df):
    pivoted_run_data_df = incoming_run_data_df.pivot_table(
        index = ["Run ID", "Time Stamp"], 
        columns = "Sensor Name",
        values = "Sensor Value"
    )
    pivoted_run_data_df = pivoted_run_data_df.reset_index()
    pivoted_run_data_df.columns
    return (pivoted_run_data_df,)


@app.cell
def _(pd, pivot_df_filled, pivoted_run_data_df):
    all_combined_df = pd.concat([pivot_df_filled, pivoted_run_data_df])
    all_combined_df_sorted = all_combined_df.sort_values(
        by=['Run ID', 'Time Stamp']
    )
    all_combined_df_sorted.columns
    return (all_combined_df_sorted,)


@app.cell
def _(all_combined_df_sorted, linregress, pd):
    identifier_cols = ["Run ID", "Time Stamp"]
    sensor_cols = [col for col in all_combined_df_sorted if col not in identifier_cols]
    def time_series_feats(group, sensor_columns):
        features = {}
        for sensor in sensor_cols:
            temp_df = group[['Time Stamp', sensor]].copy()
            temp_df.dropna(subset=[sensor], inplace = True)
            values = temp_df[sensor].dropna()
            timeStamps = temp_df["Time Stamp"]
            features[f'{sensor}_mean'] = values.mean()
            features[f'{sensor}_std_dev'] = values.std()
            features[f'{sensor}_min'] = values.min()
            features[f'{sensor}_max'] = values.max()

            if values.var() > 1e-9:
                valid_timestamps = timeStamps[values.index]
                numeric_timestamps = (valid_timestamps - valid_timestamps.iloc[0]).dt.total_seconds()
                slope, _, _, _, _ = linregress(x=numeric_timestamps, y=values)
                features[f'{sensor}_trend_slope'] = slope
            else:
                features[f'{sensor}_trend_slope'] = 0.0

        return pd.Series(features)

    run_level_features_df = all_combined_df_sorted.groupby("Run ID").apply(
        time_series_feats, sensor_columns = sensor_cols
    ).fillna(0)
    run_level_features_df.head()
    return (run_level_features_df,)


@app.cell
def _(all_run_data_df):
    static_feature_columns = ["Consumable Life", "Tool ID"]
    agg_dict_static = {col: 'first' for col in static_feature_columns}
    static_feature_df = all_run_data_df.groupby("Run ID").agg(agg_dict_static)
    static_feature_df.head()
    return (static_feature_df,)


@app.cell
def _(pd, run_level_features_df, static_feature_df):
    # Merge Consumbale life to df
    final_features_df = pd.merge(
        left = run_level_features_df,
        right = static_feature_df,
        how="left",
        left_index = True,
        right_index = True
    )
    final_features_df = final_features_df.reset_index()
    return (final_features_df,)


@app.cell
def _(load_concat):
    final_coordinates_df = load_concat("./data/train/", "metrology_data*.parquet")
    final_coordinates_df.head()
    target_df = final_coordinates_df.pivot_table(
        index = "Run ID",
        columns="Point Index",
        values="Measurement"
    )
    target_df = target_df.reset_index()
    target_df
    return (target_df,)


@app.cell
def _(final_features_df, pd, target_df):
    training_df = pd.merge(
        left = final_features_df,
        right = target_df, 
        how = "inner", 
        on = "Run ID"
    )
    training_df.shape

    return (training_df,)


@app.cell
def _(
    DataLoader,
    LabelEncoder,
    TensorDataset,
    np,
    torch,
    train_test_split,
    training_df,
):
    target_columns = [i for i in range(0, 49)]
    feature_columns = [col for col in training_df if col not in target_columns]
    feature_columns.pop(0)
    y = training_df[target_columns].values.astype(np.float32)
    X = training_df[feature_columns].values
    # Step 1: extract all Tool IDs (second-last column)
    tool_ids = [row[-1] for row in X]

    # Step 2: encode them
    le = LabelEncoder()
    encoded_tool_ids = le.fit_transform(tool_ids)

    # Step 3: overwrite second-last column in each row
    for i in range(len(X)):
        X[i][-1] = float(encoded_tool_ids[i])
    X = X.astype(np.float32)

    # --- 1b. Create Training, Validation, and Test Sets ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # --- 1c. Convert NumPy arrays to PyTorch Tensors ---
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)

    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # --- 1d. Create PyTorch DataLoaders ---
    # DataLoaders handle batching, shuffling, etc. automatically.
    BATCH_SIZE = 32

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



    return X_train, test_loader, train_loader, val_loader, y_train


@app.cell
def _(X_train, nn, torch, y_train):
    num_features = X_train.shape[1]
    num_outputs = y_train.shape[1]

    class MLP(nn.Module):
        def __init__(self, num_features, num_outputs):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                # Input Layer
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),

                # Hidden Layer 1
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),

                # Hidden Layer 2
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),

                # Output Layer
                nn.Linear(64, num_outputs)
            )

        def forward(self, x):
            return self.layers(x)

    # Instantiate the model
    pytorch_model = MLP(num_features, num_outputs)

    # Move model to GPU if available (optional but recommended)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pytorch_model.to(device)
    pytorch_model
    return device, pytorch_model


@app.cell
def _(
    device,
    mo,
    nn,
    np,
    optim,
    pytorch_model,
    test_loader,
    torch,
    train_loader,
    val_loader,
):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    EPOCHS = 50

    # --- Training Loop ---
    # Lists to store loss history for plotting
    train_losses = []
    val_losses = []


    for epoch in range(EPOCHS):
        # --- Training Phase ---
        pytorch_model.train() # Set the model to training mode (enables dropout, etc.)
        batch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Move data to the same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 1. Forward pass: compute predicted y by passing x to the model
            y_pred = pytorch_model(X_batch)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y_batch)
            batch_train_loss += loss.item() # .item() gets the scalar value of the loss

            # 3. Zero gradients: clear the gradients of all optimized variables
            optimizer.zero_grad()

            # 4. Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # 5. Update weights: call step() to cause the optimizer to update the parameters
            optimizer.step()

        # Calculate average training loss for the epoch
        avg_train_loss = batch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        pytorch_model.eval() # Set the model to evaluation mode (disables dropout, etc.)
        batch_val_loss = 0.0
        with torch.no_grad(): # In validation, we don't need to compute gradients
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = pytorch_model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                batch_val_loss += loss.item()

        # Calculate average validation loss for the epoch
        avg_val_loss = batch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print progress for each epoch
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("Training complete.")

    # --- Evaluation Loop ---
    
        # 1. Set the model to evaluation mode.
        # This is crucial as it disables layers like Dropout.
    pytorch_model.eval()
    
        # Variable to accumulate the loss
    total_test_loss = 0.0
    
        # 2. Disable gradient calculations to save memory and computation
    with torch.no_grad():
        # 3. Loop through the test data loader
        for X_batch, y_batch in test_loader:
            # Move data to the same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
            # Make predictions
            y_pred = pytorch_model(X_batch)
        
            # Calculate the loss (MSE) for this batch
            loss = loss_fn(y_pred, y_batch)
        
            # Accumulate the loss
            # We multiply by the batch size to get the total sum of squared errors,
            # which is more accurate than averaging averages if the last batch is smaller.
            total_test_loss += loss.item() * X_batch.size(0)

    # 4. Calculate the final average Mean Squared Error (MSE)
    # Divide the total loss by the total number of samples in the test set
    avg_test_mse = total_test_loss / len(test_loader.dataset)

    # 5. Calculate the Root Mean Squared Error (RMSE)
    final_rmse = np.sqrt(avg_test_mse)

    # --- Display the Results ---
    mo.md("#### Final Model Performance on Unseen Test Data:")
    mo.md(f"- **Average Test MSE**: {avg_test_mse:.6f}")
    mo.md(f"### **Final Test RMSE**: {final_rmse:.6f}")

    return


if __name__ == "__main__":
    app.run()
