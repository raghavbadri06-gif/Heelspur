# -- coding: utf-8 --
import os, random, shutil
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
from pyswarm import pso  # For PSO

# -----------------------------
# Config
# -----------------------------
ROOT = r"/nfsshare/users/raghavan/heelspurfinal/Traininghs/"
OUT_SPLIT = r"/nfsshare/users/raghavan/heelspurfinal/splitgradcam/"
RESULTS = r"/nfsshare/users/raghavan/heelspurfinal/novelswarmpart3/"
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
# 4) Model Definition
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

# -----------------------------
# 5) PSO Hyperparameter Optimization
# -----------------------------
def objective_function(params):
    lr, dropout_rate, weight_decay, focal_gamma, lambda_ce = params
    print(f"[PSO] Testing LR={lr:.6f}, Dropout={dropout_rate:.3f}, WD={weight_decay:.6f}, Gamma={focal_gamma:.2f}, Lambda_CE={lambda_ce:.2f}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    model = ConvNeXtAttentionModel(num_classes=num_classes, dropout_rate=dropout_rate).to(DEVICE)
    criterion = CombinedLoss(weight=cls_w, gamma=focal_gamma, lambda_ce=lambda_ce)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epochs = 5
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
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
        scheduler.step(val_loss)
        best_val_acc = max(best_val_acc, val_acc)
        print(f"[PSO Epoch {epoch}/{epochs}] Val Acc: {val_acc:.4f}")
    
    return -best_val_acc

# Define bounds for PSO optimization
lb = [1e-5, 0.1, 1e-6, 0.5, 0.3]  # Lower bounds: [lr, dropout, weight_decay, focal_gamma, lambda_ce]
ub = [1e-3, 0.5, 1e-3, 3.0, 0.9]  # Upper bounds: [lr, dropout, weight_decay, focal_gamma, lambda_ce]

print("[INFO] Starting PSO hyperparameter optimization...")
optimal_params, optimal_value = pso(objective_function, lb, ub, swarmsize=10, maxiter=10)
optimal_lr, optimal_dropout, optimal_wd, optimal_gamma, optimal_lambda_ce = optimal_params

print(f"[INFO] PSO found optimal parameters:")
print(f"  Learning Rate: {optimal_lr:.6f}")
print(f"  Dropout Rate: {optimal_dropout:.3f}")
print(f"  Weight Decay: {optimal_wd:.6f}")
print(f"  Focal Loss Gamma: {optimal_gamma:.2f}")
print(f"  Lambda CE: {optimal_lambda_ce:.2f}")
print(f"[INFO] Best validation accuracy: {-optimal_value:.4f}")

# Save optimal hyperparameters to CSV
hyperparams_df = pd.DataFrame({
    'Parameter': ['Learning Rate', 'Dropout Rate', 'Weight Decay', 'Focal Loss Gamma', 'Lambda CE', 'Best Validation Accuracy'],
    'Value': [optimal_lr, optimal_dropout, optimal_wd, optimal_gamma, optimal_lambda_ce, -optimal_value],
    'Description': [
        'Optimal learning rate for Adam optimizer',
        'Optimal dropout rate for regularization',
        'Optimal weight decay for L2 regularization',
        'Optimal gamma parameter for focal loss',
        'Weight for CE loss in combined loss function',
        'Best validation accuracy achieved during PSO optimization'
    ]
})
hyperparams_df.to_csv(os.path.join(RESULTS, 'optimal_hyperparameters.csv'), index=False)
print(f"[INFO] Optimal hyperparameters saved to: {os.path.join(RESULTS, 'optimal_hyperparameters.csv')}")

# Update global variables with optimal values
LR = optimal_lr
DROPOUT_RATE = optimal_dropout
WEIGHT_DECAY = optimal_wd
FOCAL_GAMMA = optimal_gamma
LAMBDA_CE = optimal_lambda_ce

# -----------------------------
# 6) Training Loop with Optimized Hyperparameters
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

model = ConvNeXtAttentionModel(num_classes=num_classes, dropout_rate=DROPOUT_RATE).to(DEVICE)
print("[INFO] ConvNeXt Tiny Attention model with Adapted CBAM ready.")

# Use the combined loss with optimal parameters
criterion = CombinedLoss(weight=cls_w, gamma=FOCAL_GAMMA, lambda_ce=LAMBDA_CE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(1, EPOCHS + 1):
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
    print(f"[Epoch {epoch}/{EPOCHS}] Train: {epoch_loss:.4f}, {epoch_acc:.4f} | Val: {val_loss:.4f}, {val_acc:.4f}")
    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
               os.path.join(RESULTS, f'checkpoint_epoch{epoch}.pth'))
torch.save(model.state_dict(), os.path.join(RESULTS, 'convnext_tiny_cbam_attention_model_final.pth'))

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

# -----------------------------
# 9) Enhanced Grad-CAM with Labels
# -----------------------------
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
save_gradcam_dir = os.path.join(RESULTS, "gradcam_results")
os.makedirs(save_gradcam_dir, exist_ok=True)

def create_gradcam_visualization(image_path, model, cam, transform, class_names, save_dir):
    img = Image.open(image_path).convert("RGB")
    original_img = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
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
    cam_image_pil = Image.fromarray(np.uint8(cam_image * 255))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')
    ax2.imshow(cam_image)
    true_label = os.path.basename(os.path.dirname(image_path))
    pred_info = f"Predicted: {class_names[pred_class]}\nConfidence: {confidence:.3f}\nTrue: {true_label}"
    text_color = 'green' if class_names[pred_class] == true_label else 'red'
    ax2.text(0.02, 0.98, pred_info, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             color=text_color, fontweight='bold')
    ax2.set_title('Grad-CAM Heatmap (Heel Spur Focus)', fontsize=12)
    ax2.axis('off')
    plt.tight_layout()
    base_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"{base_name}_gradcam.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return save_path, class_names[pred_class], true_label, confidence

gradcam_results = []
test_samples = test_dataset.samples[:]
print("Generating Grad-CAM visualizations...")
for idx, (img_path, true_label_idx) in enumerate(test_samples):
    try:
        save_path, pred_class, true_class, confidence = create_gradcam_visualization(
            img_path, model, cam, val_transforms, class_names, save_gradcam_dir
        )
        gradcam_results.append({
            'image': os.path.basename(img_path),
            'predicted': pred_class,
            'true': true_class,
            'confidence': confidence,
            'correct': pred_class == true_class,
            'visualization': os.path.basename(save_path)
        })
        print(f"Processed {idx + 1}/{len(test_samples)}: {os.path.basename(img_path)}")
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        continue

# Save Grad-CAM results to CSV
results_df = pd.DataFrame(gradcam_results)
results_df.to_csv(os.path.join(RESULTS, 'gradcam_predictions.csv'), index=False)

print(f"\n[INFO] All Grad-CAM visualizations saved in {RESULTS}")
print(f"[INFO] Results summary:")
print(f" - Correct predictions: {sum(results_df['correct'])}/{len(results_df)}")
print(f" - Accuracy: {sum(results_df['correct']) / len(results_df) * 100:.2f}%")
print(f" - Grad-CAM results saved to: {os.path.join(RESULTS, 'gradcam_predictions.csv')}")
print(f" - Optimal hyperparameters saved to: {os.path.join(RESULTS, 'optimal_hyperparameters.csv')}") This is the code sir
