import os
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your training logs
log_path = "./models/train.log"  # Update this path accordingly

def plot_metrics(log_path):
    # Load the training logs
    if not os.path.exists(log_path):
        print(f"Error: Log file '{log_path}' not found.")
        return

    # Initialize lists to store data
    learning_rates = []
    test_cers = []
    train_losses = []

    # Read log file and extract metrics
    with open(log_path, 'r') as f:
        for line in f:
            if "learning_rate" in line and "avg_test_cer" in line and "avg_train_loss" in line:
                # Extract learning rate, test CER, and train loss
                parts = line.strip().split(',')
                learning_rates.append(float(parts[0].split(':')[1].strip()))
                test_cers.append(float(parts[1].split(':')[1].strip()))
                train_losses.append(float(parts[2].split(':')[1].strip()))

    # Convert lists to NumPy arrays
    learning_rates = np.array(learning_rates)
    test_cers = np.array(test_cers)
    train_losses = np.array(train_losses)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Learning Rate Plot
    plt.subplot(3, 1, 1)
    plt.plot(learning_rates, label='Learning Rate')
    plt.title('Learning Rate over Training')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()

    # Test CER Plot
    plt.subplot(3, 1, 2)
    plt.plot(test_cers, label='Test CER')
    plt.title('Test Character Error Rate (CER) over Training')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.legend()

    # Train Loss Plot
    plt.subplot(3, 1, 3)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Train Loss over Training')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Execute the plot_metrics function
plot_metrics(log_path)
