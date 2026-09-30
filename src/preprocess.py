import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

def process_and_scale_data():
    # 1. Load pipeline configurations
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    raw_path = config["data_ingestion"]["raw_data_path"]     
    processed_path = config["data_preprocessing"]["processed_data_path"] 

    # 2. Extract arrays from the raw archive
    print(f"[STAGE 02] Loading raw data from {raw_path}...")
    raw_data = np.load(raw_path)
    x_train_raw = raw_data['x_train']
    y_train_raw = raw_data['y_train']
    x_test_raw = raw_data['x_test']
    y_test_raw = raw_data['y_test']

    # 3. Scale pixel values to [0.0, 1.0] range
    print("[STAGE 02] Normalizing pixel values...")
    x_train_scaled = x_train_raw.astype('float32') / 255.0
    x_test_scaled = x_test_raw.astype('float32') / 255.0

    # 4. Split a validation set out of the training data
    # Extracts validation parameters from config if available (defaults to 10% split with a fixed seed)
    preprocess_params = config.get("preprocessing", {})
    val_size = preprocess_params.get("val_size", 0.1)
    random_state = preprocess_params.get("random_state", 42)
    
    print(f"[STAGE 02] Splitting {val_size*100}% of training data for validation...")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_scaled, 
        y_train_raw, 
        test_size=val_size, 
        random_state=random_state,
        stratify=y_train_raw # Ensures clothing distributions match across train/val
    )

    # 5. Save the resulting train/val/test arrays under data/processed/
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    np.savez_compressed(
        processed_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test_scaled,
        y_test=y_test_raw
    )
    
    print(f"[STAGE 02] Preprocessing completed. Saved data sets to {processed_path}")
    print(f"Shapes -> Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test_scaled.shape}")

if __name__ == "__main__":
    process_and_scale_data()
