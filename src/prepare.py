import os
import numpy as np
import yaml
import tensorflow as tf

def read_and_return_data():
    # 1. Load the raw numpy arrays directly
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # 2. Extract configuration path
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    raw_path = config["data_ingestion"]["raw_data_path"]  # e.g., "data/raw/fashion_mnist.npz"

    # 3. Create target directory safely
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    # 4. Save the raw dimensions natively as a compressed npz file
    np.savez_compressed(
        raw_path, 
        x_train=x_train, 
        y_train=y_train, 
        x_test=x_test, 
        y_test=y_test
    )
    
    print(f"[STAGE 01] Ingested raw dataset successfully to {raw_path}")

if __name__ == "__main__":
    read_and_return_data()
