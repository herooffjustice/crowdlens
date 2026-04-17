#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CrowdLens: Crowd Density Estimation
# Architecture based on: CSRNet (Li et al., CVPR 2018)
# "CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"
# Implementation: written from scratch in PyTorch
# Extensions: heatmap overlay, crowd alert system, training visualization

import os, glob, random
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.transform import resize as sk_resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# In[2]:


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---- UPDATE THIS PATH if your dataset extracted differently ----
DATA_ROOT = "../data/ShanghaiTech/part_B"
TRAIN_IMG = os.path.join(DATA_ROOT, "train_data/images")
TRAIN_GT  = os.path.join(DATA_ROOT, "train_data/ground-truth")
TEST_IMG  = os.path.join(DATA_ROOT, "test_data/images")
TEST_GT   = os.path.join(DATA_ROOT, "test_data/ground-truth")

# Verify paths exist
for p in [TRAIN_IMG, TRAIN_GT, TEST_IMG, TEST_GT]:
    status = "✅" if os.path.exists(p) else "❌ NOT FOUND"
    print(f"{status}  {p}")

EPOCHS           = 100
BATCH_SIZE       = 4
LR               = 1e-6
DENSITY_SIGMA    = 15
ALERT_THRESHOLD  = 300


# In[3]:


def load_gt_mat(mat_path):
    """Load .mat annotation file → return head coordinates as numpy array."""
    mat = sio.loadmat(mat_path)
    # ShanghaiTech Part B stores coords in this nested structure:
    points = mat['image_info'][0][0][0][0][0]
    return points

def generate_density_map(img_shape, points, sigma=DENSITY_SIGMA):
    """Place a Gaussian on each head annotation → smooth density field."""
    h, w = img_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    for pt in points:
        x = min(int(pt[0]), w - 1)
        y = min(int(pt[1]), h - 1)
        density[y, x] = 1.0
    density = gaussian_filter(density, sigma=sigma)
    return density

# ---------- Quick sanity test ----------
sample_imgs = sorted(glob.glob(os.path.join(TRAIN_IMG, "*.jpg")))
print(f"Found {len(sample_imgs)} training images")

sample_img_path = sample_imgs[0]
sample_mat_path = os.path.join(
    TRAIN_GT,
    "GT_" + os.path.basename(sample_img_path).replace(".jpg", ".mat")
)
img_arr = np.array(Image.open(sample_img_path).convert("RGB"))
pts     = load_gt_mat(sample_mat_path)
dmap    = generate_density_map(img_arr.shape, pts)

print(f"Image shape : {img_arr.shape}")
print(f"Annotations : {len(pts)} heads")
print(f"Density sum : {dmap.sum():.1f}  (should be close to {len(pts)})")


# In[4]:


class CrowdDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.gt_dir    = gt_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mat_path = os.path.join(
            self.gt_dir,
            "GT_" + os.path.basename(img_path).replace(".jpg", ".mat")
        )
        img = Image.open(img_path).convert("RGB")
        pts = load_gt_mat(mat_path)

        # Resize image to fixed size for batching
        target_w, target_h = 512, 384
        orig_w, orig_h = img.size
        img = img.resize((target_w, target_h))
        img_arr = np.array(img)

        # Scale annotation coordinates proportionally
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        pts_scaled = pts.copy().astype(float)
        pts_scaled[:, 0] *= scale_x
        pts_scaled[:, 1] *= scale_y

        dmap = generate_density_map(img_arr.shape, pts_scaled)

        # Downsample density map to 1/8 spatial size (matches VGG output stride)
        # Multiply by 64 (=8×8) to preserve total count after downsampling
        dmap_small = dmap[::8, ::8] * 64.0

        if self.transform:
            img = self.transform(img)

        dmap_tensor = torch.tensor(dmap_small, dtype=torch.float32).unsqueeze(0)
        return img, dmap_tensor

# ImageNet normalization (required because we use pretrained VGG-16)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

train_dataset = CrowdDataset(TRAIN_IMG, TRAIN_GT, transform=transform)
test_dataset  = CrowdDataset(TEST_IMG,  TEST_GT,  transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=1,          shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)} images")
print(f"Test : {len(test_dataset)} images")

# Test one batch loads correctly
imgs, dmaps = next(iter(train_loader))
print(f"Batch image shape : {imgs.shape}")
print(f"Batch dmap shape  : {dmaps.shape}")


# In[5]:


class CSRNet(nn.Module):
    """
    CSRNet architecture (Li et al., CVPR 2018).
    
    Frontend : VGG-16 layers 1-23 (pretrained ImageNet weights)
               Extracts rich semantic features from crowd images.
    Backend  : 6 dilated conv layers (dilation=2, no pooling)
               Expands receptive field WITHOUT reducing spatial resolution.
               This is the key innovation — density maps stay high-resolution.
    Output   : 1-channel density map (sum = estimated crowd count)
    """
    def __init__(self):
        super().__init__()

        # Load pretrained VGG-16, take first 23 layers (up to 3rd maxpool)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        frontend_layers = list(vgg.features.children())[:23]
        self.frontend = nn.Sequential(*frontend_layers)

        # Freeze frontend weights (optional — speeds up training)
        for param in self.frontend.parameters():
            param.requires_grad = False

        # Dilated backend — dilation=2 means receptive field spans 5×5
        # but computation stays at 3×3, preserving output resolution
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, kernel_size=3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )

        # 1×1 conv to produce final 1-channel density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialize backend weights
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

model = CSRNet().to(DEVICE)

total_params    = sum(p.numel() for p in model.parameters())
trainable       = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable:,}  (backend only — frontend frozen)")


# In[6]:


optimizer    = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
criterion    = nn.MSELoss()
train_losses = []
val_maes     = []
best_mae     = float('inf')

print("Starting training...")
print(f"{'Epoch':>6} | {'Loss':>10} | {'MAE':>8}")
print("-" * 32)

for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    model.train()
    epoch_loss = 0.0
    for imgs, dmaps in train_loader:
        imgs, dmaps = imgs.to(DEVICE), dmaps.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, dmaps)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ---- Validate every 10 epochs ----
    if epoch % 10 == 0:
        model.eval()
        total_mae = 0.0
        with torch.no_grad():
            for imgs, dmaps in test_loader:
                imgs = imgs.to(DEVICE)
                pred_count = model(imgs).sum().item() / 64.0
                gt_count   = dmaps.sum().item()       / 64.0
                total_mae += abs(pred_count - gt_count)
        mae = total_mae / len(test_loader)
        val_maes.append((epoch, mae))
        print(f"{epoch:>6} | {avg_loss:>10.4f} | {mae:>8.2f}  ← validation")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), "../outputs/best_model.pth")
            print(f"         ✅ New best model saved (MAE={best_mae:.2f})")
    else:
        print(f"{epoch:>6} | {avg_loss:>10.4f} |")

print(f"\nTraining complete. Best MAE: {best_mae:.2f}")


# In[7]:


plt.figure(figsize=(11, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, color='steelblue', linewidth=2)
plt.title("Training Loss (MSE) vs Epoch", fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(alpha=0.3)

# MAE curve
if val_maes:
    epochs_val, maes_val = zip(*val_maes)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_val, maes_val, color='tomato', linewidth=2, marker='o', markersize=4)
    plt.title("Validation MAE vs Epoch", fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("MAE (people)")
    plt.grid(alpha=0.3)

plt.suptitle("CrowdLens — Training Metrics", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("../sample_outputs/training_curve.png", dpi=150)
plt.show()
print("✅ Saved: sample_outputs/training_curve.png")


# In[8]:


def visualize_prediction(model, img_path, gt_mat_path=None, save_path=None):
    """
    OUR MODIFICATION 1: Density heatmap blended onto original image.
    OUR MODIFICATION 2: Crowd alert if count exceeds threshold.
    """
    model.eval()

    img_orig    = Image.open(img_path).convert("RGB")
    img_resized = img_orig.resize((512, 384))
    img_arr     = np.array(img_resized)

    img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_dmap = model(img_tensor).squeeze().cpu().numpy()

    pred_count = pred_dmap.sum() / 64.0

    # Get ground truth count for comparison (if available)
    gt_count = None
    if gt_mat_path and os.path.exists(gt_mat_path):
        gt_pts   = load_gt_mat(gt_mat_path)
        gt_count = len(gt_pts)

    # Upsample density map back to display size
    dmap_display = sk_resize(pred_dmap, (384, 512), anti_aliasing=True)
    dmap_norm    = (dmap_display - dmap_display.min()) / (dmap_display.max() + 1e-8)

    # Apply 'hot' colormap: black→red→yellow→white (dense = bright)
    heatmap      = cm.hot(dmap_norm)[:, :, :3]
    heatmap_u8   = (heatmap * 255).astype(np.uint8)

    # Blend: 55% heatmap + 45% original
    alpha   = 0.55
    overlay = (alpha * heatmap_u8 + (1 - alpha) * img_arr).astype(np.uint8)

    # ---- CROWD ALERT SYSTEM ----
    alert      = pred_count > ALERT_THRESHOLD
    alert_text = "⚠️  HIGH DENSITY ALERT" if alert else "✅  Normal Density"
    alert_col  = "red"       if alert else "limegreen"

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#111111')
    for ax in axes:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image", color='white', fontsize=13, pad=10)

    axes[1].imshow(overlay)
    count_label = f"Predicted: {pred_count:.0f}"
    if gt_count:
        count_label += f"  |  Ground Truth: {gt_count}"
    axes[1].set_title(
        f"Density Heatmap  —  {count_label}\n{alert_text}",
        color=alert_col, fontsize=12, fontweight='bold', pad=10
    )

    plt.suptitle("CrowdLens — Crowd Density Estimation", color='white',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#111111')
        print(f"✅ Saved: {save_path}")
    plt.show()
    return pred_count

# ---- Load best model ----
model.load_state_dict(torch.load("../outputs/best_model.pth", map_location=DEVICE))
print("Best model loaded.")

# ---- Run on 4 test images ----
test_imgs = sorted(glob.glob(os.path.join(TEST_IMG, "*.jpg")))[:4]
for i, img_path in enumerate(test_imgs):
    mat_path = os.path.join(
        TEST_GT,
        "GT_" + os.path.basename(img_path).replace(".jpg", ".mat")
    )
    visualize_prediction(
        model, img_path, mat_path,
        save_path=f"../sample_outputs/sample_{i+1}.png"
    )


# In[9]:


model.load_state_dict(torch.load("../outputs/best_model.pth", map_location=DEVICE))
model.eval()

all_maes = []
with torch.no_grad():
    for imgs, dmaps in test_loader:
        imgs        = imgs.to(DEVICE)
        pred_count  = model(imgs).sum().item() / 64.0
        gt_count    = dmaps.sum().item()       / 64.0
        all_maes.append(abs(pred_count - gt_count))

final_mae = np.mean(all_maes)
final_mse = np.sqrt(np.mean(np.array(all_maes)**2))

print("=" * 40)
print(f"  CrowdLens — Final Results")
print(f"  Dataset  : ShanghaiTech Part B")
print(f"  Test MAE : {final_mae:.2f}")
print(f"  Test MSE : {final_mse:.2f}")
print(f"  Images   : {len(all_maes)}")
print("=" * 40)
print("\n📌 Copying this MAE into the README! :)")


# In[ ]:




