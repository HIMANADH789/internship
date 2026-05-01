import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

os.makedirs('results', exist_ok=True)

# Baseline Data
baseline_epochs = list(range(1, 8))
baseline_train_loss = [0.3827, 0.3474, 0.3208, 0.2798, 0.2194, 0.1547, 0.0983]
baseline_val_auc = [0.8537, 0.8674, 0.8428, 0.8453, 0.8066, 0.8089, 0.7811]

# Structure Constrained Data
struct_epochs = list(range(1, 13))
struct_train_loss = [0.4412, 0.4319, 0.4280, 0.4248, 0.4214, 0.4184, 0.4145, 0.4102, 0.4055, 0.3987, 0.3895, 0.3760]
struct_val_auc = [0.6914, 0.7224, 0.7190, 0.7104, 0.7291, 0.7219, 0.7438, 0.7239, 0.7275, 0.7195, 0.7215, 0.7144]

# Save permanent outputs to CSV
baseline_df = pd.DataFrame({'Epoch': baseline_epochs, 'Train_Loss': baseline_train_loss, 'Val_AUC': baseline_val_auc})
baseline_df.to_csv('results/baseline_training_metrics.csv', index=False)

struct_df = pd.DataFrame({'Epoch': struct_epochs, 'Train_Loss': struct_train_loss, 'Val_AUC': struct_val_auc})
struct_df.to_csv('results/structure_constrained_training_metrics.csv', index=False)

# Test AUCs
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
baseline_test_auc = [0.6902, 0.8272, 0.7014, 0.8297, 0.8655]
struct_test_auc = [0.5769, 0.6735, 0.6227, 0.7300, 0.7068]

test_df = pd.DataFrame({'Label': labels, 'Baseline_AUC': baseline_test_auc, 'Struct_Constrained_AUC': struct_test_auc})
test_df.to_csv('results/test_auc_comparison.csv', index=False)

# 1. Training Loss Comparison Plot
plt.figure(figsize=(10, 6))
plt.plot(baseline_epochs, baseline_train_loss, label='Baseline (Swin)', marker='o')
plt.plot(struct_epochs, struct_train_loss, label='Structure Constrained', marker='s')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/training_loss_comparison.png')
plt.close()

# 2. Validation AUC Comparison Plot
plt.figure(figsize=(10, 6))
plt.plot(baseline_epochs, baseline_val_auc, label='Baseline (Swin)', marker='o')
plt.plot(struct_epochs, struct_val_auc, label='Structure Constrained', marker='s')
plt.title('Validation AUC Comparison')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig('results/val_auc_comparison.png')
plt.close()

# 3. Per-Class Test AUC Comparison Plot
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, baseline_test_auc, width, label='Baseline (Swin)')
rects2 = ax.bar(x + width/2, struct_test_auc, width, label='Structure Constrained')

ax.set_ylabel('Test AUC')
ax.set_title('Per-Class Test AUC Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig('results/per_class_auc_comparison.png')
plt.close()
