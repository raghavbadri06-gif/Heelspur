# -*- coding: utf-8 -*-
import os, random, shutil
import sys
import locale
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
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pyswarm import pso

# Set encoding for the entire script
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# -----------------------------
# Custom Dataset with Unicode Handling
# -----------------------------
class UnicodeImageFolder(datasets.ImageFolder):
    """ImageFolder with robust Unicode filename handling"""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
    
    @staticmethod
    def safe_decode(text):
        """Safely decode text, handling Unicode errors"""
        try:
            return text.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            try:
                return text.encode('ascii', errors='ignore').decode('ascii')
            except:
                return str(hash(text))
    
    def find_classes(self, directory):
        """Override to handle Unicode directory names"""
        classes = []
        for entry in os.scandir(directory):
            if entry.is_dir():
                # Safely decode class name
                safe_name = self.safe_decode(entry.name)
                classes.append(safe_name)
        
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}")
        
        classes.sort()
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx
    
    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        """Override to handle Unicode filenames"""
        instances = []
        directory = os.path.expanduser(directory)
        
        if extensions is not None:
            def is_valid_file(x):
                return x.lower().endswith(extensions)
        
        # Get all class directories with original names
        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            # Find matching class index using safe comparison
            class_idx = None
            safe_class_name = self.safe_decode(class_name)
            for cls, idx in class_to_idx.items():
                if cls == safe_class_name:
                    class_idx = idx
                    break
            
            if class_idx is None:
                continue
            
            # Walk through files
            for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        instances.append((path, class_idx))
        
        return instances

# -----------------------------
# Config
# -----------------------------
ROOT = r"/nfsshare/users/raghavan/heelspurfinal/Traininghs/"
OUT_SPLIT = r"/nfsshare/users/raghavan/heelspurfinal/splitgradcam/"
RESULTS = r"/nfsshare/users/raghavan/heelspurfinal/novelswarmpart3_thadalan/"
os.makedirs(OUT_SPLIT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(os.path.join(RESULTS, 'stage_contributions'), exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42
LR = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Stage names for ConvNeXt
STAGE_NAMES = ['Stage 1 (Low-level)', 'Stage 2 (Mid-level)', 'Stage 3 (High-level)', 'Stage 4 (Semantic)']
STAGE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# -----------------------------
# 1) Stratified Split with Unicode Handling
# -----------------------------
def safe_copy(src, dst):
    """Safely copy files with Unicode names"""
    try:
        shutil.copy2(src, dst)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # If Unicode error, try with encoded filename
        try:
            # Create a safe filename
            safe_name = os.path.basename(src).encode('ascii', errors='ignore').decode('ascii')
            if not safe_name:
                safe_name = f"image_{hash(src)}.png"
            dst_dir = os.path.dirname(dst)
            safe_dst = os.path.join(dst_dir, safe_name)
            shutil.copy2(src, safe_dst)
        except:
            print(f"Warning: Could not copy {src}")

def safe_makedirs(path):
    """Safely create directories with Unicode paths"""
    try:
        os.makedirs(path, exist_ok=True)
    except UnicodeEncodeError:
        # Create with ASCII-safe path
        safe_path = path.encode('ascii', errors='ignore').decode('ascii')
        os.makedirs(safe_path, exist_ok=True)

def stratified_split_folder(in_dir, out_dir, ratios=(0.7, 0.15, 0.15), seed=SEED):
    assert abs(sum(ratios) - 1.0) < 1e-6
    
    # Get classes safely
    classes = []
    for d in os.listdir(in_dir):
        if os.path.isdir(os.path.join(in_dir, d)):
            try:
                safe_d = d.encode('utf-8', errors='ignore').decode('utf-8')
            except:
                safe_d = d.encode('ascii', errors='ignore').decode('ascii')
            classes.append((d, safe_d))  # Store both original and safe names
    
    print(f"Found classes: {[c[1] for c in classes]}")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        for _, safe_c in classes:
            split_path = os.path.join(out_dir, split, safe_c)
            safe_makedirs(split_path)
    
    # Process each class
    for orig_c, safe_c in classes:
        class_path = os.path.join(in_dir, orig_c)
        image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        files = []
        
        # Get files safely
        try:
            for f in os.listdir(class_path):
                if f.lower().endswith(image_extensions):
                    files.append(os.path.join(class_path, f))
        except:
            continue
        
        files.sort()
        
        if len(files) == 0:
            continue
            
        # Split files
        train_files, temp = train_test_split(files, test_size=(1 - ratios[0]), random_state=seed, shuffle=True)
        val_frac = ratios[1] / (ratios[1] + ratios[2])
        val_files, test_files = train_test_split(temp, test_size=(1 - val_frac), random_state=seed, shuffle=True)
        
        # Copy files
        for f in train_files:
            safe_copy(f, os.path.join(out_dir, 'train', safe_c, os.path.basename(f)))
        for f in val_files:
            safe_copy(f, os.path.join(out_dir, 'val', safe_c, os.path.basename(f)))
        for f in test_files:
            safe_copy(f, os.path.join(out_dir, 'test', safe_c, os.path.basename(f)))

if not any(Path(OUT_SPLIT).iterdir()):
    print('Creating train/val/test split...')
    stratified_split_folder(ROOT, OUT_SPLIT)
else:
    print('Split folder already exists - skipping split.')

# -----------------------------
# 2) Datasets & Augmentation (using Unicode-safe dataset)
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

# Use custom Unicode-safe dataset
try:
    train_dataset = UnicodeImageFolder(os.path.join(OUT_SPLIT, 'train'), transform=train_transforms)
    val_dataset = UnicodeImageFolder(os.path.join(OUT_SPLIT, 'val'), transform=val_transforms)
    test_dataset = UnicodeImageFolder(os.path.join(OUT_SPLIT, 'test'), transform=val_transforms)
except Exception as e:
    print(f"Error loading datasets: {e}")
    # Fallback to standard ImageFolder
    train_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT, 'val'), transform=val_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT, 'test'), transform=val_transforms)

class_names = train_dataset.classes
num_classes = len(class_names)
print('Classes:', class_names)

# -----------------------------
# 3) Handle class imbalance
# -----------------------------
train_targets = [y for _, y in train_dataset.samples]
class_counts = Counter(train_targets)
class_sample_count = np.array([class_counts.get(i, 0) for i in range(num_classes)])
class_weights_for_sampling = 1. / (class_sample_count + 1e-8)  # Add small epsilon
samples_weight = np.array([class_weights_for_sampling[t] for t in train_targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
cls_w = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_targets)
cls_w = torch.tensor(cls_w, dtype=torch.float).to(DEVICE)

# -----------------------------
# 4) Multi-Stage Model with Contribution Analysis
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

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        ce = F.cross_entropy(inputs, targets, weight=self.weight)
        return self.lambda_ce * ce + (1 - self.lambda_ce) * focal

class MultiStageConvNeXt(nn.Module):
    """
    ConvNeXt with multi-stage outputs for contribution analysis
    """
    def __init__(self, backbone_name='convnext_tiny', pretrained=True, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        # Get backbone with all stage features
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0, 
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )
        
        # Get feature dimensions
        dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            features_list = self.backbone(dummy_input)
        
        self.stage_dims = [f.shape[1] for f in features_list]
        print(f"Stage dimensions: {self.stage_dims}")
        
        # CBAM for each stage
        self.cbam_stages = nn.ModuleList([
            CBAM(dim, ratio=16) for dim in self.stage_dims
        ])
        
        # Stage classifiers
        self.stage_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            ) for dim in self.stage_dims
        ])
        
        # Attention weights for stage fusion
        total_dims = sum(self.stage_dims)
        self.stage_attention = nn.Sequential(
            nn.Linear(total_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
        )
        
        # Project all stages to common dimension
        self.fusion_dim = 512
        self.stage_projections = nn.ModuleList([
            nn.Conv2d(dim, self.fusion_dim, 1) for dim in self.stage_dims
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.fusion_dim * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x, return_stage_outputs=False):
        # Get features from all stages
        stage_features = self.backbone(x)
        
        # Apply CBAM to each stage
        attended_features = []
        for i, (feat, cbam) in enumerate(zip(stage_features, self.cbam_stages)):
            attended = cbam(feat)
            attended_features.append(attended)
        
        # Get stage-specific predictions
        stage_logits = []
        stage_confidences = []
        for i, (feat, classifier) in enumerate(zip(attended_features, self.stage_classifiers)):
            logits = classifier(feat)
            stage_logits.append(logits)
            confidence = torch.softmax(logits, dim=1).max(dim=1)[0]
            stage_confidences.append(confidence)
        
        # Calculate stage attention weights
        pooled_features = []
        for feat in attended_features:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
            pooled_features.append(pooled)
        
        attention_input = torch.cat(pooled_features, dim=1)
        stage_weights = self.stage_attention(attention_input)
        
        # Project stages to common dimension
        projected_stages = []
        for i, (feat, proj) in enumerate(zip(attended_features, self.stage_projections)):
            projected = proj(feat)
            projected_stages.append(projected)
        
        # Weighted fusion of stages
        fused_features = []
        for i, (proj, weight) in enumerate(zip(projected_stages, stage_weights.unbind(dim=1))):
            weighted = proj * weight.view(-1, 1, 1, 1)
            fused_features.append(weighted)
        
        # Concatenate all weighted features
        fusion_input = torch.cat(fused_features, dim=1)
        
        # Final classification
        final_logits = self.classifier(fusion_input)
        
        if return_stage_outputs:
            return final_logits, stage_logits, stage_weights, stage_confidences
        
        return final_logits

# -----------------------------
# 5) PSO Hyperparameter Optimization
# -----------------------------
def objective_function(params):
    lr, dropout_rate, weight_decay, focal_gamma, lambda_ce = params
    print(f"[PSO] Testing LR={lr:.6f}, Dropout={dropout_rate:.3f}, WD={weight_decay:.6f}, Gamma={focal_gamma:.2f}, Lambda_CE={lambda_ce:.2f}")
    
    try:
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
        
        model = MultiStageConvNeXt(num_classes=num_classes, dropout_rate=dropout_rate).to(DEVICE)
        criterion = CombinedLoss(weight=cls_w, gamma=focal_gamma, lambda_ce=lambda_ce)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        epochs = 3
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            model.train()
            running_acc = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                final_logits, stage_logits, stage_weights, stage_conf = model(imgs, return_stage_outputs=True)
                
                loss_main = criterion(final_logits, labels)
                loss_aux = sum([criterion(logit, labels) for logit in stage_logits]) / len(stage_logits)
                loss = loss_main + 0.3 * loss_aux
                
                loss.backward()
                optimizer.step()
                running_acc += (final_logits.argmax(1) == labels).sum().item()
            
            model.eval()
            val_acc = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    final_logits = model(imgs)
                    val_acc += (final_logits.argmax(1) == labels).sum().item()
            
            val_acc /= len(val_loader.dataset)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"[PSO Epoch {epoch}/{epochs}] Val Acc: {val_acc:.4f}")
        
        return -best_val_acc
    
    except Exception as e:
        print(f"PSO iteration failed: {e}")
        return 1.0  # Return worst possible value

# Define bounds for PSO optimization
lb = [1e-5, 0.1, 1e-6, 0.5, 0.3]
ub = [1e-3, 0.5, 1e-3, 3.0, 0.9]

print("[INFO] Starting PSO hyperparameter optimization...")
try:
    optimal_params, optimal_value = pso(objective_function, lb, ub, swarmsize=5, maxiter=5)
    optimal_lr, optimal_dropout, optimal_wd, optimal_gamma, optimal_lambda_ce = optimal_params
except Exception as e:
    print(f"PSO failed: {e}")
    print("Using default parameters")
    optimal_lr, optimal_dropout, optimal_wd, optimal_gamma, optimal_lambda_ce = 1e-4, 0.3, 1e-5, 2.0, 0.7

print(f"[INFO] Using parameters:")
print(f"  Learning Rate: {optimal_lr:.6f}")
print(f"  Dropout Rate: {optimal_dropout:.3f}")
print(f"  Weight Decay: {optimal_wd:.6f}")
print(f"  Focal Loss Gamma: {optimal_gamma:.2f}")
print(f"  Lambda CE: {optimal_lambda_ce:.2f}")

# Update global variables
LR = optimal_lr
DROPOUT_RATE = optimal_dropout
WEIGHT_DECAY = optimal_wd
FOCAL_GAMMA = optimal_gamma
LAMBDA_CE = optimal_lambda_ce

# -----------------------------
# 6) Full Training
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

model = MultiStageConvNeXt(num_classes=num_classes, dropout_rate=DROPOUT_RATE).to(DEVICE)
print("[INFO] Multi-Stage ConvNeXt with CBAM ready.")

criterion = CombinedLoss(weight=cls_w, gamma=FOCAL_GAMMA, lambda_ce=LAMBDA_CE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        final_logits, stage_logits, stage_weights, stage_conf = model(imgs, return_stage_outputs=True)
        
        loss_main = criterion(final_logits, labels)
        loss_aux = sum([criterion(logit, labels) for logit in stage_logits]) / len(stage_logits)
        loss = loss_main + 0.3 * loss_aux
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        running_acc += (final_logits.argmax(1) == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    
    # Validation
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            final_logits = model(imgs)
            loss = criterion(final_logits, labels)
            val_loss += loss.item() * imgs.size(0)
            val_acc += (final_logits.argmax(1) == labels).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step(val_loss)
    print(f"[Epoch {epoch}/{EPOCHS}] Train: {epoch_loss:.4f}, {epoch_acc:.4f} | Val: {val_loss:.4f}, {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(RESULTS, 'multistage_model_best.pth'))

# -----------------------------
# 7) Test Evaluation
# -----------------------------
model.eval()
y_true, y_pred, y_prob = [], [], []
all_stage_weights = []
all_stage_confidences = []
all_stage_preds = []

print("\n[INFO] Collecting stage contributions on test set...")

with torch.no_grad():
    for idx, (imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        final_logits, stage_logits, stage_weights, stage_conf = model(imgs, return_stage_outputs=True)
        probs = torch.softmax(final_logits, dim=1)
        
        y_true.append(labels.item())
        y_pred.append(final_logits.argmax(1).item())
        y_prob.append(probs.cpu().numpy()[0])
        
        all_stage_weights.append(stage_weights[0].cpu().numpy())
        all_stage_confidences.append([sc[0].item() for sc in stage_conf])
        all_stage_preds.append([sl.argmax(1).item() for sl in stage_logits])

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)
stage_weights_array = np.array(all_stage_weights)
stage_confidences_array = np.array(all_stage_confidences)
stage_preds_array = np.array(all_stage_preds)

# Calculate stage accuracies
stage_accuracies = [(stage_preds_array[:, i] == y_true).mean() for i in range(4)]

# -----------------------------
# 8) Save Results
# -----------------------------
# Stage analysis CSV
stage_df = pd.DataFrame({
    'true_class': y_true,
    'true_class_name': [class_names[t] for t in y_true],
    'final_pred': y_pred,
    'final_pred_name': [class_names[p] for p in y_pred],
    'stage1_weight': stage_weights_array[:, 0],
    'stage2_weight': stage_weights_array[:, 1],
    'stage3_weight': stage_weights_array[:, 2],
    'stage4_weight': stage_weights_array[:, 3],
    'stage1_confidence': stage_confidences_array[:, 0],
    'stage2_confidence': stage_confidences_array[:, 1],
    'stage3_confidence': stage_confidences_array[:, 2],
    'stage4_confidence': stage_confidences_array[:, 3],
    'stage1_pred': stage_preds_array[:, 0],
    'stage2_pred': stage_preds_array[:, 1],
    'stage3_pred': stage_preds_array[:, 2],
    'stage4_pred': stage_preds_array[:, 3],
    'final_correct': (y_pred == y_true).astype(int)
})

stage_df.to_csv(os.path.join(RESULTS, 'stage_contributions', 'stage_analysis.csv'), index=False, encoding='utf-8-sig')

# Stage statistics
stage_stats = pd.DataFrame({
    'Stage': STAGE_NAMES,
    'Mean Weight': stage_weights_array.mean(axis=0),
    'Std Weight': stage_weights_array.std(axis=0),
    'Accuracy': stage_accuracies,
    'Mean Confidence': stage_confidences_array.mean(axis=0)
})
stage_stats.to_csv(os.path.join(RESULTS, 'stage_contributions', 'stage_statistics.csv'), index=False, encoding='utf-8-sig')

# Plot stage weights
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(STAGE_NAMES))
bars = plt.bar(x_pos, stage_weights_array.mean(axis=0), yerr=stage_weights_array.std(axis=0),
               capsize=5, color=STAGE_COLORS, alpha=0.8, edgecolor='black')
plt.xlabel('Network Stages', fontsize=12)
plt.ylabel('Attention Weight', fontsize=12)
plt.title('Stage-wise Contribution to Final Prediction', fontsize=14)
plt.xticks(x_pos, STAGE_NAMES, rotation=45, ha='right')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

for bar, mean in zip(bars, stage_weights_array.mean(axis=0)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'stage_contributions', 'stage_weights.png'), dpi=150, bbox_inches='tight')
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(RESULTS, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS, 'classification_report.csv'), encoding='utf-8-sig')

# ROC Curves
plt.figure(figsize=(8, 6))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# -----------------------------
# 9) Final Summary
# -----------------------------
print("\n" + "="*80)
print("Multi-Stage ConvNeXt Training Complete")
print("="*80)
print(f"\nModel Performance:")
print(f"  Test Accuracy: {np.mean(y_true == y_pred):.4f}")
print(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")

print(f"\nStage Statistics:")
for i, name in enumerate(STAGE_NAMES):
    print(f"\n  {name}:")
    print(f"    Mean Weight: {stage_weights_array[:, i].mean():.4f} ± {stage_weights_array[:, i].std():.4f}")
    print(f"    Accuracy: {stage_accuracies[i]:.4f}")
    print(f"    Mean Confidence: {stage_confidences_array[:, i].mean():.4f}")

print(f"\nResults saved to: {RESULTS}")
print("="*80) 
