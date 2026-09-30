import os
import numpy as np
import yaml

def process_and_scale_data():
    # 1. Load pipeline configurations
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    raw_path = config["data_ingestion"]["raw_data_path"]     # e.g., "data/raw/fashion_mnist.npz"
    processed_path = config["data_preprocessing"]["processed_data_path"] # e.g., "data/processed/fashion_mnist_scaled.npz"

    # 2. Extract arrays from the raw archive
    print(f"[STAGE 02] Loading raw data from {raw_path}...")
    raw_data = np.load(raw_path)
    x_train = raw_data['x_train']
    y_train = raw_data['y_train']
    x_test = raw_data['x_test']
    y_test = raw_data['y_test']

    # 3. Scale pixel values to [0.0, 1.0] range
    # Converting to float32 prevents precision issues and prepares it natively for Keras/TensorFlow
    x_train_scaled = x_train.astype('float32') / 255.0
    x_test_scaled = x_test.astype('float32') / 255.0

    # 4. Save processed arrays back into a compressed archive
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    np.savez_compressed(
        processed_path,
        x_train=x_train_scaled,
        y_train=y_train,
        x_test=x_test_scaled,
        y_test=y_test
    )
    
    print(f"[STAGE 02] Preprocessing completed. Scaled data saved to {processed_path}")

if __name__ == "__main__":
    process_and_scale_data()
