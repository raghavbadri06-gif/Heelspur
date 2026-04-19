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
import torch.nn.functional as F  # Added for adaptive pooling
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# -----------------------------
# Config
# -----------------------------
ROOT = r"/nfsshare/users/raghavan/Heel Spur 29th september/Traininghs/"
OUT_SPLIT = r"/nfsshare/users/raghavan/Heel Spur 29th september/splitgradcam/"
RESULTS = r"/nfsshare/users/raghavan/heelspurfinal/novel7thoctober/"
os.makedirs(OUT_SPLIT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42
LR = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 1) Stratified Split
# -----------------------------
def stratified_split_folder(in_dir, out_dir, ratios=(0.7,0.15,0.15), seed=SEED):
    assert abs(sum(ratios)-1.0) < 1e-6
    classes = [d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,d))]
    print(f"Found classes: {classes}")

    for split in ["train","val","test"]:
        for c in classes:
            os.makedirs(os.path.join(out_dir, split, c), exist_ok=True)

    for c in classes:
        class_path = os.path.join(in_dir, c)
        image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith(image_extensions)]
        files.sort()

        train_files, temp = train_test_split(files, test_size=(1-ratios[0]), random_state=seed, shuffle=True)
        val_frac = ratios[1] / (ratios[1]+ratios[2])
        val_files, test_files = train_test_split(temp, test_size=(1-val_frac), random_state=seed, shuffle=True)

        for f in train_files: shutil.copy(f, os.path.join(out_dir,'train',c, os.path.basename(f)))
        for f in val_files: shutil.copy(f, os.path.join(out_dir,'val',c, os.path.basename(f)))
        for f in test_files: shutil.copy(f, os.path.join(out_dir,'test',c, os.path.basename(f)))

if not any(Path(OUT_SPLIT).iterdir()):
    print('Creating train/val/test split...')
    stratified_split_folder(ROOT, OUT_SPLIT)
else:
    print('Split folder already exists - skipping split.')

# -----------------------------
# 2) Datasets & Augmentation
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8,1.0)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT,'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT,'val'), transform=val_transforms)
test_dataset = datasets.ImageFolder(os.path.join(OUT_SPLIT,'test'), transform=val_transforms)
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
cls_w = torch.tensor(cls_w,dtype=torch.float).to(DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# -----------------------------
# 4) Novel Enhancement: Enhanced Efficient Attention with CBAM
# 
# Key Enhancements:
# - Use backbone.forward_features() to access the spatial feature map (B, C, H, W) from the last stage.
# - Integrate Convolutional Block Attention Module (CBAM) for channel-wise and spatial attention on the feature map.
#   This allows the model to dynamically focus on important channels and spatial regions, improving interpretability
#   and performance on medical images like heel spurs where subtle features matter.
# - Retain the original global feature attention for fine-grained gating on pooled features.
# - This creates a hierarchical attention mechanism: spatial/channel on maps -> global on pooled features.
# - Novelty: Hybrid CNN attention pyramid that refines multi-resolution features before classification,
#   tailored for fine-grained medical diagnostics without adding excessive parameters.
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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
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

class EnhancedEfficientAttentionModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True, num_classes=3):
        super().__init__()
        # Backbone configured to enable feature map access
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, features_only=True)
        
        # Get feature dimension from last stage
        dummy_input = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            features_list = self.backbone(dummy_input)
            last_features = features_list[-1]  # Last stage feature map
        feature_dim = last_features.shape[1]
        
        # CBAM for spatial and channel attention on feature maps
        self.cbam = CBAM(feature_dim, ratio=16)
        
        # Global attention on pooled features (original mechanism)
        self.global_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract multi-stage features
        features_list = self.backbone(x)
        feat_map = features_list[-1]  # Use last stage for detailed spatial info
        
        # Apply CBAM attention to feature map
        attended_map = self.cbam(feat_map)
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(attended_map, (1, 1)).flatten(1)
        
        # Apply global attention
        attention_weights = self.global_attention(pooled_features)
        attended_features = pooled_features * attention_weights
        
        # Classify
        output = self.classifier(attended_features)
        return output

# Use the enhanced model
model = EnhancedEfficientAttentionModel(num_classes=num_classes).to(DEVICE)
print("[INFO] Enhanced Efficient Attention model with CBAM ready.")

criterion = nn.CrossEntropyLoss(weight=cls_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# -----------------------------
# 5) Training Loop
# -----------------------------
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(1, EPOCHS+1):
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
        running_acc += (outputs.argmax(1)==labels).sum().item()

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
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_acc += (outputs.argmax(1)==labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    scheduler.step(val_loss)
    print(f"[Epoch {epoch}/{EPOCHS}] Train: {epoch_loss:.4f}, {epoch_acc:.4f} | Val: {val_loss:.4f}, {val_acc:.4f}")

    torch.save({'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict()},
               os.path.join(RESULTS,f'checkpoint_epoch{epoch}.pth'))

torch.save(model.state_dict(), os.path.join(RESULTS,'enhanced_cbam_attention_model_final.pth'))

# -----------------------------
# 6) Plots: Accuracy & Loss
# -----------------------------
plt.figure()
plt.plot(range(1,EPOCHS+1), train_accs, label='train')
plt.plot(range(1,EPOCHS+1), val_accs, label='val')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(RESULTS,'accuracy.png')); plt.close()

plt.figure()
plt.plot(range(1,EPOCHS+1), train_losses, label='train')
plt.plot(range(1,EPOCHS+1), val_losses, label='val')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(RESULTS,'loss.png')); plt.close()

# -----------------------------
# 7) Test Evaluation: Confusion + ROC
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

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.savefig(os.path.join(RESULTS,'confusion_matrix.png')); plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
pd.DataFrame(report).transpose().to_csv(os.path.join(RESULTS,'classification_report.csv'))

# ROC Curves
plt.figure(figsize=(8,6))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true==i, y_prob[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves'); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(RESULTS,'roc_curves.png')); plt.close()

# -----------------------------
# 8) Enhanced Grad-CAM with Labels
# 
# Note: Grad-CAM target remains on backbone layers for visualization of pre-attention activations.
# The CBAM enhances training/inference but Grad-CAM focuses on backbone for interpretability.
# -----------------------------
def find_target_layer(model):
    # Use the last convolutional layer in the backbone
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d) and 'block' in name:
            target_layer = module
    return target_layer

target_layer = find_target_layer(model)
cam = GradCAM(model=model, target_layers=[target_layer])

save_gradcam_dir = os.path.join(RESULTS, "gradcam_results")
os.makedirs(save_gradcam_dir, exist_ok=True)

def create_gradcam_visualization(image_path, model, cam, transform, class_names, save_dir):
    """Create Grad-CAM visualization with labels"""
    # Load and process image
    img = Image.open(image_path).convert("RGB")
    original_img = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = outputs.argmax(1).item()
        confidence = probs[0][pred_class].item()
    
    # Generate Grad-CAM
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    
    # Prepare image for overlay
    rgb_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.float32(rgb_img) / 255.0
    
    # Create heatmap
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Convert to PIL for drawing
    cam_image_pil = Image.fromarray(np.uint8(cam_image * 255))
    
    # Create figure with labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=12)
    ax1.axis('off')
    
    # Grad-CAM image with prediction info
    ax2.imshow(cam_image)
    
    # Add prediction information as text
    true_label = os.path.basename(os.path.dirname(image_path))
    pred_info = f"Predicted: {class_names[pred_class]}\nConfidence: {confidence:.3f}\nTrue: {true_label}"
    
    # Color text based on correctness
    text_color = 'green' if class_names[pred_class] == true_label else 'red'
    ax2.text(0.02, 0.98, pred_info, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             color=text_color, fontweight='bold')
    
    ax2.set_title('Grad-CAM Visualization', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    base_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"{base_name}_gradcam.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return save_path, class_names[pred_class], true_label, confidence

# Process test images for Grad-CAM
gradcam_results = []
test_samples = test_dataset.samples[:]  # all

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
        
        print(f"Processed {idx+1}/{len(test_samples)}: {os.path.basename(img_path)}")
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        continue

# Save results to CSV
results_df = pd.DataFrame(gradcam_results)
results_df.to_csv(os.path.join(RESULTS, 'gradcam_predictions.csv'), index=False)

# Enhanced HTML Visualization
html_file = os.path.join(RESULTS, 'visualize_gradcam.html')
with open(html_file, 'w') as f:
    f.write('''
    <html>
    <head>
        <title>Grad-CAM Visualization Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { display: flex; flex-wrap: wrap; gap: 20px; }
            .card { 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 10px; 
                width: 300px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card img { width: 100%; height: auto; }
            .correct { border: 2px solid green; }
            .incorrect { border: 2px solid red; }
            .info { padding: 10px; }
            .confidence { font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Grad-CAM Visualization Results</h1>
        <div class="container">
    ''')
    
    for result in gradcam_results:
        card_class = "correct" if result['correct'] else "incorrect"
        f.write(f'''
            <div class="card {card_class}">
                <img src="gradcam_results/{result['visualization']}" alt="{result['image']}">
                <div class="info">
                    <p><strong>Image:</strong> {result['image']}</p>
                    <p><strong>Predicted:</strong> {result['predicted']}</p>
                    <p><strong>True:</strong> {result['true']}</p>
                    <p class="confidence">Confidence: {result['confidence']:.3f}</p>
                    <p><strong>Status:</strong> {"Correct" if result['correct'] else "Incorrect"}</p>
                </div>
            </div>
        ''')
    
    f.write('''
        </div>
    </body>
    </html>
    ''')

print(f"\n[INFO] All Grad-CAM visualizations saved in {RESULTS}")
print(f"[INFO] Results summary:")
print(f"  - Correct predictions: {sum(results_df['correct'])}/{len(results_df)}")
print(f"  - Accuracy: {sum(results_df['correct'])/len(results_df)*100:.2f}%")
print(f"  - HTML report: {html_file}") 
