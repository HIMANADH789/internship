import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

results_dir = r'c:\internship\Altaf\results'
os.makedirs(results_dir, exist_ok=True)

# Training Loss Data
epochs = [0, 1]
train_loss = [0.08518149775161517, 0.0309569947135869]

# Test Metrics Data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [0.8920628834355828, 0.9154540893125941, 0.9416774193548387, 0.9283806131535428]

# Save to CSV
loss_df = pd.DataFrame({'Epoch': epochs, 'Train_Loss': train_loss})
loss_df.to_csv(os.path.join(results_dir, 'training_metrics.csv'), index=False)

metrics_df = pd.DataFrame({'Metric': metrics, 'Value': values})
metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)

# Plot 1: Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='b', label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'training_loss.png'))
plt.close()

# Plot 2: Test Metrics
plt.figure(figsize=(10, 6))
sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis')
plt.title('Final Test Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0.8, 1.0)
for i, v in enumerate(values):
    plt.text(i, v + 0.002, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'test_metrics.png'))
plt.close()
