pip install timm einops torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

import kagglehub

# Download latest version of CheXpert
path = kagglehub.dataset_download("ashery/chexpert")

print("Path to dataset files:", path)


import os

os.listdir("/kaggle/input/chexpert")



import os
DATA_ROOT = "/kaggle/input/chexpert"
TRAIN_ROOT = "/kaggle/input/chexpert/train"
VALID_ROOT = "/kaggle/input/chexpert/valid"
os.listdir(TRAIN_ROOT)[:5]




import os

patient = os.listdir(TRAIN_ROOT)[0]
study = os.listdir(os.path.join(TRAIN_ROOT, patient))[0]
files = os.listdir(os.path.join(TRAIN_ROOT, patient, study))

print(patient, study, files)



from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class CheXpertUnsupervised(Dataset):
    def __init__(self, root, transform):
        self.paths = []
        for root_, _, files in os.walk(root):
            for f in files:
                if f.endswith(".jpg"):
                    self.paths.append(os.path.join(root_, f))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


TRAIN_ROOT = "/kaggle/input/chexpert/train"

dataset = CheXpertUnsupervised(
    root=TRAIN_ROOT,
    transform=transform
)

print("Total images:", len(dataset))


loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)


images = next(iter(loader))   # [B, 3, 224, 224]
images = images.cuda() if torch.cuda.is_available() else images


import torch.nn.functional as F

# Unfold into patches
patches = F.unfold(
    images,
    kernel_size=16,
    stride=16
)
# shape: [B, 3*16*16, 196]

patches = patches.transpose(1, 2)
# shape: [B, 196, 768]


import torch

grid_size = 14
coords = []

for i in range(grid_size):
    for j in range(grid_size):
        coords.append([i, j])

coords = torch.tensor(coords, dtype=torch.float32)  # [196, 2]
coords = coords.to(images.device)


diff = coords[:, None, :] - coords[None, :, :]
dist = torch.norm(diff, dim=-1)  # [196, 196]


sigma = 2.0  # hyperparameter
P_ij = torch.exp(-dist**2 / (2 * sigma**2))


patch_feats = patches[0]   # take 1 image → [196, 768]
patch_feats = torch.nn.functional.normalize(patch_feats, dim=-1)


S_ij = patch_feats @ patch_feats.T   # [196, 196]


# simple thresholded co-activation
activation = (patch_feats.mean(dim=-1) > 0).float()  # [196]


C_ij = activation[:, None] * activation[None, :]


from torchvision import transforms

aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0)
])

images_aug = aug(images)


patches_aug = F.unfold(
    images_aug,
    kernel_size=16,
    stride=16
).transpose(1, 2)

patch_feats_aug = torch.nn.functional.normalize(patches_aug[0], dim=-1)


S_ij_aug = patch_feats_aug @ patch_feats_aug.T
T_ij = 1.0 - torch.abs(S_ij - S_ij_aug)


alpha, beta, gamma, delta = 0.25, 0.25, 0.25, 0.25

A_ij = (
    alpha * S_ij +
    beta  * P_ij +
    gamma * C_ij +
    delta * T_ij
)

A_ij = torch.clamp(A_ij, 0, 1)


import numpy as np

A_ij_np = A_ij.detach().cpu().numpy()
np.save("A_ij.npy", A_ij_np)

print("Saved A_ij.npy with shape:", A_ij_np.shape)


import matplotlib.pyplot as plt

plt.imshow(A_ij.detach().cpu())
plt.colorbar()
plt.title("Learned Structural Feasibility A_ij")
plt.show()
A_ij[:5, :5]



import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from PIL import Image
import timm
from torch.amp import autocast, GradScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


IMAGE_ROOT = "/kaggle/input/chexpert"

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

NUM_LABELS = len(LABELS)

EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 16
LR = 2e-5
WEIGHT_DECAY = 1e-4


DATA_DIR = "/kaggle/input/chexpert"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
valid_df = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))


def preprocess_labels(df):
    df = df.copy()
    df[LABELS] = df[LABELS].fillna(0)
    df[LABELS] = df[LABELS].replace(-1, 0)
    return df

train_df = preprocess_labels(train_df)
valid_df = preprocess_labels(valid_df)


# IMPORTANT: keep paths relative to /kaggle/input/chexpert
train_df["Path"] = train_df["Path"].str.replace(
    "CheXpert-v1.0-small/", "", regex=False
)
valid_df["Path"] = valid_df["Path"].str.replace(
    "CheXpert-v1.0-small/", "", regex=False
)


train_df_sub, test_df = train_test_split(
    train_df, test_size=0.1, random_state=42
)

# Use 20% of training subset (same spirit as Swin baseline)
train_df_sub = train_df_sub.sample(frac=0.2, random_state=42)

print(
    "Train:", len(train_df_sub),
    "Val:", len(valid_df),
    "Test:", len(test_df)
)


from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-base-patch4-window7-224"
)


class CheXpertDataset(Dataset):
    def __init__(self, df, image_root, processor):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_root, row["Path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        labels = torch.from_numpy(
          row[LABELS].to_numpy(dtype=np.float32)
          )


        return pixel_values, labels


train_loader = DataLoader(
    CheXpertDataset(train_df_sub, IMAGE_ROOT, processor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    CheXpertDataset(valid_df, IMAGE_ROOT, processor),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    CheXpertDataset(test_df, IMAGE_ROOT, processor),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


from google.colab import drive
drive.mount('/content/drive')


x, y = train_loader.dataset[0]
print(x.shape, y)
A_ij = np.load("A_ij.npy")           # shape [196, 196]
A_ij = torch.tensor(A_ij).float().to(device)

print("Loaded A_ij:", A_ij.shape)


class ViTWithAij(nn.Module):
    def __init__(self, num_labels, A_ij):
        super().__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_labels
        )
        self.A_ij = A_ij  # [196, 196]

    def forward(self, x):
        # Patch embedding
        x = self.vit.patch_embed(x)          # [B, 196, D]
        B, N, D = x.shape

        # Apply structural feasibility
        B = x.size(0)
        A = self.A_ij.unsqueeze(0).expand(B, -1, -1)  # [B, 196, 196]
        x = torch.bmm(A, x) # [B, 196, D]

        # CLS token
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)
        cls_out = x[:, 0]

        logits = self.vit.head(cls_out)
        return logits


model = ViTWithAij(NUM_LABELS, A_ij).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scaler = GradScaler()

best_auc = 0.0
patience_counter = 0
best_model_wts = None

for epoch in range(EPOCHS):

    # ===== TRAIN =====
    model.train()
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ✅ FIXED autocast
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ✅ FIXED autocast
            with autocast("cuda"):
                outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    auc = roc_auc_score(targets, preds, average="macro")

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val AUC: {auc:.4f}"
    )

    # ===== EARLY STOPPING =====
    if auc > best_auc:
        best_auc = auc
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break


CHECKPOINT_PATH = "vit_base_Aij_best.pt"
torch.save(best_model_wts, CHECKPOINT_PATH)

print("Saved best Model C to:", CHECKPOINT_PATH)


model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)

        preds.append(probs.cpu().numpy())
        targets.append(labels.cpu().numpy())

preds = np.concatenate(preds)
targets = np.concatenate(targets)

print("Test Macro AUC:", roc_auc_score(targets, preds, average="macro"))

print("\nPer-class Test AUC:")
for i, label in enumerate(LABELS):
    print(
        f"{label:20s}: "
        f"{roc_auc_score(targets[:, i], preds[:, i]):.4f}"
    )


def prediction_stability(model, image, trials=5, noise_std=0.01):
    model.eval()
    outputs = []

    with torch.no_grad():
        for _ in range(trials):
            noisy = image + noise_std * torch.randn_like(image)
            logits = model(noisy.unsqueeze(0).to(device))
            outputs.append(torch.sigmoid(logits).cpu().numpy())

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
        image, _ = dataset[i]
        stabilities.append(
            prediction_stability(
                model, image, trials, noise_std
            )
        )
    return float(np.mean(stabilities))

test_stability = test_prediction_stability(
    model=model,
    dataset=test_loader.dataset
)

print(f"🧠 Test Prediction Stability (↓ better): {test_stability:.6f}")
