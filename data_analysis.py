import re

def analyze_data(log_file='training_logs.log'):
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()

        # Extract and analyze relevant metrics (learning rate, CER, train loss)
        learning_rate_matches = re.findall(r'Learning rate: ([\d.]+)', log_content)
        cer_matches = re.findall(r'CER: ([\d.]+)', log_content)
        train_loss_matches = re.findall(r'Train loss: ([\d.]+)', log_content)

        # Example: Calculate average learning rate, CER, and train loss
        avg_learning_rate = sum(map(float, learning_rate_matches)) / len(learning_rate_matches) if learning_rate_matches else None
        avg_cer = sum(map(float, cer_matches)) / len(cer_matches) if cer_matches else None
        avg_train_loss = sum(map(float, train_loss_matches)) / len(train_loss_matches) if train_loss_matches else None

        print(f"Average Learning Rate: {avg_learning_rate}")
        print(f"Average CER: {avg_cer}")
        print(f"Average Train Loss: {avg_train_loss}")

    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == '__main__':
    # Add argparse for the log file path if needed
    analyze_data()
