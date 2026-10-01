import os
import json
import numpy as np
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model():
    # 1. Load pipeline configurations
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_path = config["data_preprocessing"]["processed_data_path"] # e.g., "data/processed/fashion_mnist_scaled.npz"
    model_path = "models/model.h5"
    
    # Define outputs requested at project root and assets directory
    metrics_path = "metrics.json"
    matrix_img_dir = "reports"
    os.makedirs(matrix_img_dir, exist_ok=True)
    matrix_img_path = os.path.join(matrix_img_dir, "confusion_matrix.png")

    # 2. Extract testing sets
    print(f"[STAGE 04] Loading processed test set from {processed_path}...")
    data = np.load(processed_path)
    x_test = data['x_test']
    y_test = data['y_test']

    # 3. Load the pre-trained neural network
    print(f"[STAGE 04] Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 4. Compute overall test loss and accuracy metric scores
    print("[STAGE 04] Calculating evaluation metrics on test data...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # 5. Predict labels and compile the Confusion Matrix
    y_pred_probabilities = model.predict(x_test, verbose=0)
    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)
    cm = confusion_matrix(y_test, y_pred_labels)

    # 6. Generate and save the Confusion Matrix graph image
    print(f"[STAGE 04] Drawing and saving confusion matrix to {matrix_img_path}...")
    fashion_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=fashion_labels, 
        yticklabels=fashion_labels
    )
    plt.title('Fashion MNIST Evaluation - Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    plt.savefig(matrix_img_path, dpi=300)
    plt.close()

    # 7. Write metrics out to JSON at the project root
    metrics_data = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc)
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"[STAGE 04] Success! Metrics written to {metrics_path}")

if __name__ == "__main__":
    evaluate_model()
