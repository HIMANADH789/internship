import os, time, copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())


import kagglehub

DATA_DIR = kagglehub.dataset_download("ashery/chexpert")
print("Dataset path:", DATA_DIR)


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
valid_df = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))

def preprocess_labels(df):
    df = df.copy()
    df[LABELS] = df[LABELS].fillna(0)
    df[LABELS] = df[LABELS].replace(-1, 0)
    return df

train_df = preprocess_labels(train_df)
valid_df = preprocess_labels(valid_df)

train_df["Path"] = train_df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)
valid_df["Path"] = valid_df["Path"].str.replace("CheXpert-v1.0-small/", "", regex=False)


from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-base-patch4-window7-224"
)

class CheXpertDataset(Dataset):
    def __init__(self, df, root_dir, processor):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # ✅ CRITICAL FIX
        labels = torch.from_numpy(
            row[LABELS].to_numpy(dtype=np.float32)
        )

        return pixel_values, labels



from sklearn.model_selection import train_test_split

train_df_sub, test_df = train_test_split(
    train_df, test_size=0.1, random_state=42
)

train_df_sub = train_df_sub.sample(frac=0.2, random_state=42)  # 20% baseline


train_loader = DataLoader(
    CheXpertDataset(train_df_sub, DATA_DIR, processor),
    batch_size=16, shuffle=True, num_workers=2, pin_memory=True
)

val_loader = DataLoader(
    CheXpertDataset(valid_df, DATA_DIR, processor),
    batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)

test_loader = DataLoader(
    CheXpertDataset(test_df, DATA_DIR, processor),
    batch_size=16, shuffle=False, num_workers=2, pin_memory=True
)


from transformers import SwinForImageClassification

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    num_labels=5,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
scaler = GradScaler()


from google.colab import drive
drive.mount("/content/drive")

SAVE_DIR = "/content/drive/MyDrive/CheXpert_Project"
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(SAVE_DIR, "swin_chexpert_baseline.pt")

REPORT_FILE_PATH = os.path.join(SAVE_DIR, "training_report.md")
PLOTS_DIR = os.path.join(SAVE_DIR, "report_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def write_report_markdown(message):
    with open(REPORT_FILE_PATH, "a") as f:
        f.write(message + "\n")

def add_plot_to_report(fig, plot_filename, plot_title):
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    fig.savefig(plot_path)
    relative_plot_path = os.path.join("report_plots", plot_filename) # Path relative to SAVE_DIR for Markdown
    write_report_markdown(f"\n### {plot_title}\n")
    write_report_markdown(f"![{plot_title}]({relative_plot_path})\n")
    plt.close(fig) # Close the figure to free up memory

import matplotlib.pyplot as plt
import seaborn as sns

EPOCHS, PATIENCE = 15, 5
best_auc, patience_counter = 0.0, 0

epoch_times = []
train_loss_history = [] # New: Store training loss per epoch
val_auc_history = []    # New: Store validation AUC per epoch
train_start = time.time()

write_report_markdown("# CheXpert Model Training Report\n")
write_report_markdown(f"## Training Start Time: {time.ctime(train_start)}\n")
write_report_markdown("## Training Parameters\n")
write_report_markdown(f"- Epochs: {EPOCHS}\n")
write_report_markdown(f"- Patience for Early Stopping: {PATIENCE}\n")
write_report_markdown(f"- Batch Size: {train_loader.batch_size}\n")
write_report_markdown(f"- Optimizer: {type(optimizer).__name__}\n")
write_report_markdown(f"- Learning Rate: {optimizer.param_groups[0]['lr']}\n")
write_report_markdown(f"- Device: {device}\n")

write_report_markdown("## Epoch-wise Training Progress\n")
write_report_markdown("| Epoch | Train Loss | Val AUC | Time (min) | Status |\n")
write_report_markdown("|-------|------------|---------|------------|--------|\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_start = time.time()
    train_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            loss = model(pixel_values=imgs, labels=labels).loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_loss_history.append(train_loss) # New: Store train loss

    # Validation
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(pixel_values=imgs).logits
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds, targets = np.concatenate(preds), np.concatenate(targets)
    auc = roc_auc_score(targets, preds, average="macro")
    val_auc_history.append(auc) # New: Store validation AUC

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    status_message = ""
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        patience_counter = 0
        status_message = "✅ Saved best model"
    else:
        patience_counter += 1
        status_message = f"Patience: {patience_counter}/{PATIENCE}"

    report_line = f"| {epoch+1:<5} | {train_loss:.4f}   | {auc:.4f}  | {epoch_time/60:.2f}     | {status_message} |"
    write_report_markdown(report_line)
    print(f"Epoch {epoch+1} | Loss {train_loss:.4f} | AUC {auc:.4f} | Time {epoch_time/60:.2f} min | {status_message}")

    if patience_counter >= PATIENCE:
        write_report_markdown("\n⏹️ Early stopping\n")
        print("⏹️ Early stopping")
        break

total_train_time = (time.time()-train_start)/60
write_report_markdown(f"\n## Training Summary\n")
write_report_markdown(f"- Total training time: {total_train_time:.2f} min\n")
write_report_markdown(f"- Best Validation AUC: {best_auc:.4f}\n")
print(f"Total training time: {total_train_time:.2f} min")

write_report_markdown("\n## Training Metrics Plots\n")

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
add_plot_to_report(plt.gcf(), 'training_loss.png', 'Training Loss Over Epochs')

# Plot Validation AUC
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(val_auc_history) + 1), val_auc_history, label='Validation AUC', color='orange')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Validation AUC Over Epochs')
plt.grid(True)
add_plot_to_report(plt.gcf(), 'validation_auc.png', 'Validation AUC Over Epochs')

# Load the best model after training completes
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

preds, targets = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(pixel_values=imgs).logits
        preds.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())

preds, targets = np.concatenate(preds), np.concatenate(targets)

test_auc = roc_auc_score(targets, preds, average="macro")
write_report_markdown(f"\n- **Final Test Macro AUC:** {test_auc:.4f}\n")
print("Final Test Macro AUC:", test_auc)

write_report_markdown("\n### Per-class Test AUC\n")
print("\nPer-class Test AUC:")

per_class_auc = []
for i, label in enumerate(LABELS):
    auc_i = roc_auc_score(targets[:, i], preds[:, i])
    per_class_auc.append({'Label': label, 'AUC': auc_i})
    write_report_markdown(f"- {label:20s}: {auc_i:.4f}")
    print(f"{label:20s}: {auc_i:.4f}")

# Plot Per-class AUC
per_class_auc_df = pd.DataFrame(per_class_auc)
plt.figure(figsize=(12, 6))
sns.barplot(x='Label', y='AUC', data=per_class_auc_df, palette='viridis')
plt.xlabel('Disease Label')
plt.ylabel('AUC Score')
plt.title('Per-class AUC on Test Set')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
add_plot_to_report(plt.gcf(), 'per_class_auc.png', 'Per-class AUC on Test Set')

def prediction_stability(model, image, trials=5, noise_std=0.01):
    model.eval()
    outputs = []

    with torch.no_grad():  # ✅ CRITICAL FIX
        for _ in range(trials):
            noisy = image + noise_std * torch.randn_like(image)
            logits = model(
                pixel_values=noisy.unsqueeze(0).to(device)
            ).logits

            outputs.append(
                torch.sigmoid(logits).detach().cpu().numpy()
            )

    outputs = np.array(outputs)
    return outputs.var(axis=0).mean()

def test_prediction_stability(
    model,
    dataset,
    num_samples=50,
    trials=5,
    noise_std=0.01
):
    stabilities = []

    for i in range(num_samples):
        image, _ = dataset[i]  # label not needed
        stability = prediction_stability(
            model=model,
            image=image,
            trials=trials,
            noise_std=noise_std
        )
        stabilities.append(stability)

    return float(np.mean(stabilities))

test_stability = test_prediction_stability(
    model=model,
    dataset=test_loader.dataset,
    num_samples=50,
    trials=5,
    noise_std=0.01
)
write_report_markdown(f"\n- **Test Prediction Stability (↓ better):** {test_stability:.6f}\n")
print(f"\n🧠 Test Prediction Stability (↓ better): {test_stability:.6f}")

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

preds, targets = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(pixel_values=imgs).logits
        preds.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())

preds, targets = np.concatenate(preds), np.concatenate(targets)

print("Test AUC:", roc_auc_score(targets, preds, average="macro"))


print("\nPer-class Test AUC:")
for i, label in enumerate(LABELS):
    auc_i = roc_auc_score(targets[:, i], preds[:, i])
    print(f"{label:20s}: {auc_i:.4f}")


def prediction_stability(model, image, trials=5, noise_std=0.01):
    model.eval()
    outputs = []

    with torch.no_grad():  # ✅ CRITICAL FIX
        for _ in range(trials):
            noisy = image + noise_std * torch.randn_like(image)
            logits = model(
                pixel_values=noisy.unsqueeze(0).to(device)
            ).logits

            outputs.append(
                torch.sigmoid(logits).detach().cpu().numpy()
            )

    outputs = np.array(outputs)
    return outputs.var(axis=0).mean()


def test_prediction_stability(
    model,
    dataset,
    num_samples=50,
    trials=5,
    noise_std=0.01
):
    stabilities = []

    for i in range(num_samples):
        image, _ = dataset[i]  # label not needed
        stability = prediction_stability(
            model=model,
            image=image,
            trials=trials,
            noise_std=noise_std
        )
        stabilities.append(stability)

    return float(np.mean(stabilities))


test_stability = test_prediction_stability(
    model=model,
    dataset=test_loader.dataset,  # IMPORTANT
    num_samples=50,
    trials=5,
    noise_std=0.01
)

print(f"🧠 Test Prediction Stability (↓ better): {test_stability:.6f}")


print(
    "Train:", len(train_loader.dataset),
    "Val:", len(val_loader.dataset),
    "Test:", len(test_loader.dataset)
)
