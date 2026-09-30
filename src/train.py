import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

def train_model():
    # 1. Load pipeline configurations
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve paths based on standard DVC project layout
    processed_path = config["data_preprocessing"]["processed_data_path"] # e.g., "data/processed/fashion_mnist_scaled.npz"
    model_dir = config["model_dir"]  # e.g., "models/ann_model"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model.h5")
    history_path = os.path.join(model_dir, "history.csv")

    # 2. Extract arrays from the scaled archive
    print(f"[STAGE 03] Loading scaled training data from {processed_path}...")
    data = np.load(processed_path)
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    # Inside src/train.py step 2:
    x_val = data['x_val']
    y_val = data['y_val']


    # 3. Extract hyperparameters from params.yaml if specified, or use robust defaults
    # (Checking configuration structure to keep pipeline dynamic)
    train_params = config.get("train", {})
    epochs = train_params.get("epochs", 10)
    batch_size = train_params.get("batch_size", 32)
    dropout_rate = train_params.get("dropout_rate", 0.2)
    dense_units = train_params.get("dense_units", 128)

    # 4. Build the Sequential ANN architecture
    print("[STAGE 03] Constructing Sequential ANN model architecture...")
    model = Sequential([
        Flatten(input_shape=(28, 28)),          # Flattens 2D matrices cleanly
        Dense(dense_units, activation='relu'),   # Hidden layer with ReLU activation
        Dropout(dropout_rate),                   # Regularization layer
        Dense(10, activation='softmax')          # Output layer for 10 fashion classes
    ])

    # 5. Compile the network
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 6. Train the model
    print(f"[STAGE 03] Training network for {epochs} epochs...")
    history = model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val), # Changed from (x_test, y_test)
    verbose=1
)

    # 7. Save model and export history metrics
    model.save(model_path)
    print(f"[STAGE 03] Trained network model exported successfully to {model_path}")
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path, index=False)
    print(f"[STAGE 03] Training run history metrics saved to {history_path}")

if __name__ == "__main__":
    train_model()
