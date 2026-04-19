#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, random, shutil, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# OCR imports
import pytesseract
import re

# For PSO
import pyswarm as ps

# -----------------------------
# Config
# -----------------------------
ROOT = r"/nfsshare/users/raghavan/heelspurfinal/Traininghs/"
OUT_SPLIT = r"/nfsshare/users/raghavan/heelspurfinal/splitgradcam/"
RESULTS = r"/nfsshare/users/raghavan/heelspurfinal/novelswarmpart3_ocrreadcorrectedthada5_final"
os.makedirs(OUT_SPLIT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42
LR = 1e-4  # Will be optimized by PSO
NUM_WORKERS = 4
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Track timing and computation costs
start_time = time.time()
epoch_times = []
computation_costs = []

# -----------------------------
# 1) Stratified Split
# -----------------------------
def stratified_split_folder(in_dir, out_dir, ratios=(0.7, 0.15, 0.15), seed=SEED):
    assert abs(sum(ratios) - 1.0) < 1e-6
    classes = [d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]
    print(f"Found classes: {classes}")
    for split in ["train", "val", "test"]:
        for c in classes:
            os.makedirs(os.path.join(out_dir, split, c), exist_ok=True)
    for c in classes:
        class_path = os.path.join(in_dir, c)
        image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(image_extensions)]
        files.sort()
        train_files, temp = train_test_split(files, test_size=(1 - ratios[0]), random_state=seed, shuffle=True)
        val_frac = ratios[1] / (ratios[1] + ratios[2])
        val_files, test_files = train_test_split(temp, test_size=(1 - val_frac), random_state=seed, shuffle=True)
        for f in train_files:
            shutil.copy(f, os.path.join(out_dir, 'train', c, os.path.basename(f)))
        for f in val_files:
            shutil.copy(f, os.path.join(out_dir, 'val', c, os.path.basename(f)))
        for f in test_files:
            shutil.copy(f, os.path.join(out_dir, 'test', c, os.path.basename(f)))

if not any(Path(OUT_SPLIT).iterdir()):
    print('Creating train/val/test split...')
    stratified_split_folder(ROOT, OUT_SPLIT)
else:
    print('Split folder already exists - skipping split.')

# -----------------------------
# 2) Datasets & Augmentation
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT, 'val'), transform=val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT, 'test'), transform=val_transforms)
class_names = train_dataset.classes
num_classes = len(class_names)
print('Classes:', class_names)

# -----------------------------
# 3) Handle class imbalance
# -----------------------------
train_targets = [y for _, y in train_dataset.imgs]
class_counts = Counter(train_targets)
class_sample_count = np.array([class_counts[i] for i in range(num_classes)])
class_weights_for_sampling = 1. / class_sample_count
samples_weight = np.array([class_weights_for_sampling[t] for t in train_targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
cls_w = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_targets)
cls_w = torch.tensor(cls_w, dtype=torch.float).to(DEVICE)

# -----------------------------
# 4) Model Definition (same as before)
# -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x_concat)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, lambda_ce=0.7):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(weight=weight, gamma=gamma)
        self.lambda_ce = lambda_ce
        self.weight = weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        ce = F.cross_entropy(inputs, targets, weight=self.weight)
        return self.lambda_ce * ce + (1 - self.lambda_ce) * focal

class ConvNeXtAttentionModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', pretrained=True, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, features_only=True)
        dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            features_list = self.backbone(dummy_input)
            last_features = features_list[-1]
        feature_dim = last_features.shape[1]
        self.cbam = CBAM(feature_dim, ratio=16)
        self.global_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features_list = self.backbone(x)
        feat_map = features_list[-1]
        attended_map = self.cbam(feat_map)
        pooled_features = F.adaptive_avg_pool2d(attended_map, (1, 1)).flatten(1)
        attention_weights = self.global_attention(pooled_features)
        attended_features = pooled_features * attention_weights
        output = self.classifier(attended_features)
        return output

# =====================================================================
# 5) Particle Swarm Optimization (PSO) for Hyperparameter Tuning
# =====================================================================

# Create data loaders for PSO
pso_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
pso_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Store PSO history
pso_history = []
pso_best_params = {}
pso_best_score = -float('inf')

def train_evaluate(params):
    """
    Training and evaluation function for PSO
    params: [lr, dropout_rate, weight_decay, focal_gamma, lambda_ce]
    Returns negative accuracy (since pyswarm minimizes)
    """
    lr = params[0]
    dropout_rate = params[1]
    weight_decay = params[2]
    focal_gamma = params[3]
    lambda_ce = params[4]
    
    print(f"\n{'='*60}")
    print(f"[PSO] Testing parameters:")
    print(f"  Learning Rate: {lr:.6f}")
    print(f"  Dropout Rate: {dropout_rate:.3f}")
    print(f"  Weight Decay: {weight_decay:.6f}")
    print(f"  Focal Gamma: {focal_gamma:.2f}")
    print(f"  Lambda CE: {lambda_ce:.2f}")
    print(f"{'='*60}")

    model = ConvNeXtAttentionModel(num_classes=num_classes, dropout_rate=dropout_rate).to(DEVICE)
    criterion = CombinedLoss(weight=cls_w, gamma=focal_gamma, lambda_ce=lambda_ce)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    epochs = 5
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for imgs, labels in pso_train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_acc += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(pso_train_loader.dataset)
        epoch_acc = running_acc / len(pso_train_loader.dataset)

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in pso_val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(pso_val_loader.dataset)
        val_acc /= len(pso_val_loader.dataset)
        scheduler.step(val_loss)
        
        best_val_acc = max(best_val_acc, val_acc)
        
        print(f"  Epoch {epoch}/{epochs} - Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

    pso_history.append({
        'params': params,
        'accuracy': best_val_acc,
        'lr': lr,
        'dropout_rate': dropout_rate,
        'weight_decay': weight_decay,
        'focal_gamma': focal_gamma,
        'lambda_ce': lambda_ce
    })
    
    print(f"\n[PSO] Best validation accuracy: {best_val_acc:.4f}")
    
    global pso_best_score, pso_best_params
    if best_val_acc > pso_best_score:
        pso_best_score = best_val_acc
        pso_best_params = {
            'lr': lr,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay,
            'focal_gamma': focal_gamma,
            'lambda_ce': lambda_ce
        }
        print(f"[PSO] ? New best found! Accuracy: {best_val_acc:.4f}")
    
    return -best_val_acc

lb = [1e-5, 0.1, 1e-6, 0.5, 0.3]
ub = [1e-3, 0.5, 1e-3, 3.0, 0.9]

options = {
    'c1': 1.5,
    'c2': 1.5,
    'w': 0.7,
    'maxiter': 20,
    'swarmsize': 10
}

print("\n" + "="*70)
print("PARTICLE SWARM OPTIMIZATION (PSO) FOR HYPERPARAMETER TUNING")
print("="*70)
print(f"Parameter bounds:")
print(f"  Learning Rate: [{lb[0]:.6f}, {ub[0]:.6f}]")
print(f"  Dropout Rate: [{lb[1]:.1f}, {ub[1]:.1f}]")
print(f"  Weight Decay: [{lb[2]:.6f}, {ub[2]:.6f}]")
print(f"  Focal Gamma: [{lb[3]:.1f}, {ub[3]:.1f}]")
print(f"  Lambda CE: [{lb[4]:.1f}, {ub[4]:.1f}]")
print(f"PSO options: {options}")
print("="*70 + "\n")

pso_start_time = time.time()
try:
    optimal_params, optimal_value = ps.pso(
        train_evaluate, 
        lb, 
        ub, 
        swarmsize=options['swarmsize'],
        maxiter=options['maxiter'],
        omega=options['w'],
        phip=options['c1'],
        phig=options['c2']
    )
    
    optimal_accuracy = -optimal_value
    pso_time = time.time() - pso_start_time
    
    print("\n" + "="*70)
    print("PSO OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Optimal parameters found:")
    print(f"  Learning Rate: {optimal_params[0]:.6f}")
    print(f"  Dropout Rate: {optimal_params[1]:.3f}")
    print(f"  Weight Decay: {optimal_params[2]:.6f}")
    print(f"  Focal Loss Gamma: {optimal_params[3]:.2f}")
    print(f"  Lambda CE: {optimal_params[4]:.2f}")
    print(f"  Best validation accuracy: {optimal_accuracy:.4f}")
    print(f"  PSO optimization time: {pso_time:.2f} seconds")
    print("="*70)

except Exception as e:
    print(f"\n[ERROR] PSO optimization failed: {e}")
    if pso_best_params:
        optimal_params = [
            pso_best_params['lr'],
            pso_best_params['dropout_rate'],
            pso_best_params['weight_decay'],
            pso_best_params['focal_gamma'],
            pso_best_params['lambda_ce']
        ]
        optimal_accuracy = pso_best_score
        pso_time = time.time() - pso_start_time
        print(f"Using best parameters from history with accuracy: {optimal_accuracy:.4f}")
    else:
        print("No valid parameters found. Using defaults.")
        optimal_params = [1e-4, 0.3, 1e-5, 2.0, 0.7]
        optimal_accuracy = 0.0
        pso_time = 0

pso_history_df = pd.DataFrame(pso_history)
pso_history_df.to_csv(os.path.join(RESULTS, 'pso_optimization_history.csv'), index=False)

LR = optimal_params[0]
DROPOUT_RATE = optimal_params[1]
WEIGHT_DECAY = optimal_params[2]
FOCAL_GAMMA = optimal_params[3]
LAMBDA_CE = optimal_params[4]

hyperparams_df = pd.DataFrame({
    'Parameter': ['Learning Rate', 'Dropout Rate', 'Weight Decay', 'Focal Loss Gamma', 'Lambda CE', 'Best Validation Accuracy', 'PSO Time (seconds)'],
    'Value': [LR, DROPOUT_RATE, WEIGHT_DECAY, FOCAL_GAMMA, LAMBDA_CE, optimal_accuracy, pso_time],
    'Description': [
        'Optimal learning rate for Adam optimizer',
        'Optimal dropout rate for regularization',
        'Optimal weight decay for L2 regularization',
        'Optimal gamma parameter for focal loss',
        'Weight for CE loss in combined loss function',
        'Best validation accuracy achieved during PSO optimization',
        'Time taken for PSO optimization'
    ]
})
hyperparams_df.to_csv(os.path.join(RESULTS, 'optimal_hyperparameters_pso.csv'), index=False)
print(f"[INFO] Optimal hyperparameters saved")

# -----------------------------
# 6) Training Loop with Optimized Hyperparameters
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model = ConvNeXtAttentionModel(num_classes=num_classes, dropout_rate=DROPOUT_RATE).to(DEVICE)
print("[INFO] ConvNeXt Tiny Attention model ready.")

criterion = CombinedLoss(weight=cls_w, gamma=FOCAL_GAMMA, lambda_ce=LAMBDA_CE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        running_acc += (outputs.argmax(1) == labels).sum().item()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_acc += (outputs.argmax(1) == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    scheduler.step(val_loss)
    
    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)
    
    print(f"[Epoch {epoch}/{EPOCHS}] Train: {epoch_loss:.4f}, {epoch_acc:.4f} | Val: {val_loss:.4f}, {val_acc:.4f} | Time: {epoch_time:.2f}s")
    torch.save({'epoch': epoch, 'model_state': model.state_dict()},
               os.path.join(RESULTS, f'checkpoint_epoch{epoch}.pth'))
torch.save(model.state_dict(), os.path.join(RESULTS, 'model_final.pth'))

total_training_time = sum(epoch_times)
computation_costs.append({
    'total_training_time_seconds': total_training_time,
    'average_epoch_time_seconds': total_training_time / EPOCHS,
    'total_epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'total_batches': len(train_loader)
})

# -----------------------------
# 7) Plots: Accuracy & Loss
# -----------------------------
plt.figure()
plt.plot(range(1, EPOCHS + 1), train_accs, label='train')
plt.plot(range(1, EPOCHS + 1), val_accs, label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS, 'accuracy.png'))
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS + 1), train_losses, label='train')
plt.plot(range(1, EPOCHS + 1), val_losses, label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS, 'loss.png'))
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS + 1), epoch_times, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Training Time per Epoch')
plt.grid(True)
plt.savefig(os.path.join(RESULTS, 'epoch_times.png'))
plt.close()

# -----------------------------
# 8) Test Evaluation: Confusion + ROC
# -----------------------------
model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        y_true.append(labels.item())
        y_pred.append(outputs.argmax(1).item())
        y_prob.append(probs.cpu().numpy()[0])
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(RESULTS, 'confusion_matrix.png'))
plt.close()

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS, 'classification_report.csv'))

plt.figure(figsize=(8, 6))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS, 'roc_curves.png'))
plt.close()

# =====================================================================
# 9) MINIMAL Metadata Extraction using OCR (Publication-Ready)
# =====================================================================
def extract_field(text, pattern, default):
    """Extract field from OCR text using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else default

def extract_minimal_metadata_from_image(image_path):
    """
    Extract MINIMAL metadata from image using OCR.
    ONLY extracts patient ID and study description to avoid reviewer concerns
    about metadata leakage in model training.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return get_minimal_fallback_metadata(image_path)
        
        height, width = img.shape[:2]
        top_region = img[0:int(height*0.3), 0:width]
        
        gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        ocr_texts = []
        text1 = pytesseract.image_to_string(gray, config='--psm 6')
        ocr_texts.append(text1)
        text2 = pytesseract.image_to_string(thresh, config='--psm 6')
        ocr_texts.append(text2)
        
        full_text = max(ocr_texts, key=len)
        full_text = re.sub(r'\s+', ' ', full_text)
        
        # Extract ONLY minimal metadata fields
        metadata = {
            'patient_id': extract_field(full_text, r'(?:ID|Patient ID|MRN)[:\s]*([A-Z0-9-]+)', 'N/A'),
            'study_description': extract_field(full_text, r'(?:Study|Description)[:\s]*([A-Za-z0-9\s]+?)(?:\n|$)', 'Foot X-Ray'),
        }
        
        return metadata
        
    except Exception as e:
        print(f"OCR Error for {os.path.basename(image_path)}: {e}")
        return get_minimal_fallback_metadata(image_path)

def get_minimal_fallback_metadata(image_path):
    """Minimal fallback metadata when OCR fails. No DICOM parameters."""
    filename = os.path.basename(image_path)
    numbers = re.findall(r'\d+', filename)
    patient_id = numbers[0] if numbers else 'N/A'
    
    return {
        'patient_id': patient_id,
        'study_description': 'Foot X-Ray',
    }

# =====================================================================
# 10) Branch Contribution Analyzer
# =====================================================================
class BranchContributionAnalyzer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activation_maps = {}
        self.gradient_maps = {}
        
    def register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.activation_maps[name] = output.detach()
            return hook
            
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradient_maps[name] = grad_output[0].detach()
            return hook
        
        hooks = []
        for name, module in self.model.backbone.named_modules():
            if 'stages' in name and isinstance(module, nn.Sequential) and len(name.split('.')) <= 2:
                stage_name = name.replace('.', '_')
                hooks.append(module.register_forward_hook(get_activation(stage_name)))
                hooks.append(module.register_full_backward_hook(get_gradient(stage_name)))
        return hooks
    
    def compute_branch_contributions(self, image_tensor, target_class=None, normalize=True):
        hooks = self.register_hooks()
        
        if not hooks:
            return {}, target_class if target_class is not None else 0
        
        self.model.zero_grad()
        output = self.model(image_tensor)
        
        if target_class is None:
            target_class = output.argmax(1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        raw_contributions = {}
        for name, activation in self.activation_maps.items():
            if name in self.gradient_maps:
                gradients = self.gradient_maps[name]
                pos_gradients = F.relu(gradients)
                pooled_gradients = torch.mean(pos_gradients, dim=[0, 2, 3], keepdim=True)
                weighted_activation = activation * pooled_gradients
                contribution_score = weighted_activation.mean().item()
                raw_contributions[name] = contribution_score
        
        for hook in hooks:
            hook.remove()
        
        if normalize and sum(raw_contributions.values()) > 0:
            total = sum(raw_contributions.values())
            normalized = {k: v/total for k, v in raw_contributions.items()}
            return normalized, target_class
        return raw_contributions, target_class

# =====================================================================
# 11) Publication-Ready Grad-CAM Visualization
# =====================================================================
def find_target_layer(model):
    for name, module in reversed(list(model.backbone.named_modules())):
        if isinstance(module, nn.Conv2d) and 'stages.3' in name:
            return module
    for name, module in reversed(list(model.backbone.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None

target_layer = find_target_layer(model)
cam = GradCAM(model=model, target_layers=[target_layer])

save_viz_dir = os.path.join(RESULTS, "gradcam_visualizations")
os.makedirs(save_viz_dir, exist_ok=True)

branch_analyzer = BranchContributionAnalyzer(model, DEVICE)

def create_publication_gradcam(image_path, model, cam, transform, class_names, metadata, save_dir, figsize=(14, 7)):
    """
    Creates a publication-ready Grad-CAM visualization with minimal clinical context.
    No DICOM metadata to avoid reviewer concerns about training data leakage.
    """
    img = Image.open(image_path).convert("RGB")
    original_img = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    true_label = os.path.basename(os.path.dirname(image_path))

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
        confidence = probs[0][pred_class].item()

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

    rgb_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.float32(rgb_img) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # LEFT: Original Image with MINIMAL clinical context
    ax1.imshow(original_img)
    ax1.set_title('Input X-ray with clinical context', fontsize=14, fontweight='bold', pad=10)
    
    # MINIMAL metadata - only patient ID and study
    metadata_text = (
        f"Patient ID: {metadata['patient_id']}\n"
        f"Study: {metadata['study_description']}"
    )
    
    ax1.text(0.05, 0.95, metadata_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', linespacing=1.5,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, 
                      edgecolor='gray', linewidth=1, pad=8))
    ax1.axis('off')

    # RIGHT: Grad-CAM with clean prediction display
    ax2.imshow(cam_image)
    
    is_correct = class_names[pred_class] == true_label
    
    # Clean prediction display - no [CORRECT] bracket
    if is_correct:
        pred_info = (f"Predicted: {class_names[pred_class]}\n"
                     f"Confidence: {confidence:.3f}\n"
                     f"True: {true_label}\n"
                     f"? Correct prediction")
    else:
        pred_info = (f"Predicted: {class_names[pred_class]}\n"
                     f"Confidence: {confidence:.3f}\n"
                     f"True: {true_label}\n"
                     f"? Incorrect prediction")
    
    ax2.text(0.05, 0.95, pred_info, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', linespacing=1.5,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                      edgecolor='green' if is_correct else 'red', linewidth=2),
             color='black', fontweight='bold')
    
    ax2.set_title('Grad-CAM activation map', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')

    # NO suptitle - clean figure without overarching title
    plt.tight_layout()
    
    base_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"{base_name}_gradcam.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    plt.close()
    
    return save_path, class_names[pred_class], true_label, confidence, is_correct

# =====================================================================
# 12) Process all test samples with publication-ready visualization
# =====================================================================
print("\n[INFO] Generating publication-ready Grad-CAM visualizations...")
viz_results = []
all_contributions = []

for idx, (img_path, true_label_idx) in enumerate(test_dataset.samples):
    try:
        img = Image.open(img_path).convert("RGB")
        img_tensor = val_transforms(img).unsqueeze(0).to(DEVICE)

        contributions, pred_class = branch_analyzer.compute_branch_contributions(img_tensor)
        
        true_class_name = class_names[true_label_idx]
        pred_class_name = class_names[pred_class]
        
        # Extract MINIMAL metadata from this specific image
        metadata = extract_minimal_metadata_from_image(img_path)

        save_path, pred_class_name, true_class_name, confidence, is_correct = create_publication_gradcam(
            img_path, model, cam, val_transforms, class_names, metadata, save_viz_dir
        )

        # Store minimal results
        viz_results.append({
            'image': os.path.basename(img_path),
            'patient_id': metadata['patient_id'],
            'predicted': pred_class_name,
            'true': true_class_name,
            'confidence': confidence,
            'correct': is_correct,
            'visualization': os.path.basename(save_path)
        })
        
        for stage, score in contributions.items():
            clean_name = stage.replace('stages_', 'stage_')
            all_contributions.append({
                'image': os.path.basename(img_path),
                'patient_id': metadata['patient_id'],
                'true_class': true_class_name,
                'predicted_class': pred_class_name,
                'correct': is_correct,
                'confidence': confidence,
                'stage': clean_name,
                'contribution_score': score
            })
        
        status = "?" if is_correct else "?"
        print(f"[{idx+1:3d}/{len(test_dataset)}] {status} {os.path.basename(img_path):30s} "
              f"True: {true_class_name:10s} Pred: {pred_class_name:10s} Conf: {confidence:.3f}")

    except Exception as e:
        print(f"[ERROR] {os.path.basename(img_path)}: {str(e)}")
        continue

# =====================================================================
# 13) Save Results with Computation Metrics
# =====================================================================
total_time = time.time() - start_time

viz_df = pd.DataFrame(viz_results)
viz_df.to_csv(os.path.join(RESULTS, 'gradcam_results.csv'), index=False)

contrib_df = pd.DataFrame(all_contributions)
if not contrib_df.empty:
    contrib_df.to_csv(os.path.join(RESULTS, 'branch_contributions.csv'), index=False)

# Save computation costs
computation_df = pd.DataFrame(computation_costs)
computation_df.to_csv(os.path.join(RESULTS, 'computation_costs.csv'), index=False)

# Save overall metrics
overall_metrics = {
    'total_runtime_seconds': total_time,
    'total_runtime_hours': total_time / 3600,
    'pso_optimization_time_seconds': pso_time,
    'total_training_time_seconds': total_training_time,
    'average_epoch_time_seconds': total_training_time / EPOCHS,
    'total_epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'total_batches_training': len(train_loader),
    'test_samples': len(test_dataset),
    'overall_accuracy_percent': (viz_df['correct'].sum() / len(viz_df)) * 100 if len(viz_df) > 0 else 0
}

overall_metrics_df = pd.DataFrame([overall_metrics])
overall_metrics_df.to_csv(os.path.join(RESULTS, 'overall_metrics.csv'), index=False)

print(f"\n[INFO] Visualizations saved in: {save_viz_dir}")
print(f"[INFO] Results saved to CSV files")
print(f"\n[INFO] Total computation time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if not contrib_df.empty:
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    accuracy = (viz_df['correct'].sum() / len(viz_df)) * 100
    print(f"\n? Overall Test Accuracy: {accuracy:.2f}% ({viz_df['correct'].sum()}/{len(viz_df)})")
    print(f"\n? Computation Summary:")
    print(f"  - Total Runtime: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"  - PSO Optimization: {pso_time:.2f} seconds")
    print(f"  - Training Time: {total_training_time:.2f} seconds")
    print(f"  - Average Epoch Time: {total_training_time/EPOCHS:.2f} seconds")
    print(f"  - Total Epochs: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Total Training Batches: {len(train_loader)}")

print("\n? Processing complete! Publication-ready Grad-CAM visualizations generated.")
