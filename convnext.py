# -*- coding: windows-1252 -*-
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, balanced_accuracy_score
from sklearn.utils import resample
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Config
# -----------------------------
ROOT = r"/nfsshare/users/raghavan/heelspurfinal/Traininghs/"
OUT_SPLIT = r"/nfsshare/users/raghavan/heelspurfinal/splitgradcam/"
RESULTS = r"/nfsshare/users/raghavan/heelspurfinal/convnext_2026results/"
os.makedirs(OUT_SPLIT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
SEED = 42
LR = 1e-4
NUM_WORKERS = 4
CONFIDENCE_THRESHOLD = 0.7  # Threshold for high-confidence predictions
DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Results directory: {RESULTS}")

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
# 4) ConvNeXt Tiny Model
# -----------------------------
class ConvNextModel(nn.Module):
    def __init__(self, backbone_name='convnext_tiny', pretrained=True, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        return self.backbone(x)

model = ConvNextModel(num_classes=num_classes).to(DEVICE)
print("[INFO] ConvNeXt Tiny model ready.")

criterion = nn.CrossEntropyLoss(weight=cls_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# -----------------------------
# 5) Training Loop (Simplified - No confidence tracking)
# -----------------------------
train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0

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
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(RESULTS, 'best_model.pth'))
    
    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    torch.save({'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict()},
               os.path.join(RESULTS, f'checkpoint_epoch{epoch}.pth'))

torch.save(model.state_dict(), os.path.join(RESULTS, 'convnext_model_final.pth'))

# -----------------------------
# 6) Plots: Accuracy & Loss
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1,EPOCHS+1), train_accs, label='Train', marker='o')
plt.plot(range(1,EPOCHS+1), val_accs, label='Validation', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Training & Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(1,EPOCHS+1), train_losses, label='Train', marker='o')
plt.plot(range(1,EPOCHS+1), val_losses, label='Validation', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training & Validation Loss')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'training_curves.png'), dpi=150)
plt.close()

# -----------------------------
# 7) Test Evaluation with Confidence Scores
# -----------------------------
model.eval()
y_true, y_pred, y_prob, confidences = [], [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        
        y_true.append(labels.item())
        y_pred.append(pred.item())
        y_prob.append(probs.cpu().numpy()[0])
        confidences.append(confidence.item())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)
confidences = np.array(confidences)

# Calculate metrics with confidence intervals
def bootstrap_confidence_interval(scores, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence interval"""
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(scores, n_samples=len(scores))
        bootstrap_scores.append(np.mean(bootstrap_sample))
    
    lower = np.percentile(bootstrap_scores, (100-ci)/2)
    upper = np.percentile(bootstrap_scores, 100 - (100-ci)/2)
    return lower, upper, np.mean(bootstrap_scores)

# Calculate accuracy with CI
accuracy = np.mean(y_true == y_pred)
acc_lower, acc_upper, acc_bootstrap = bootstrap_confidence_interval(y_true == y_pred)

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# High confidence predictions analysis
high_conf_mask = confidences >= CONFIDENCE_THRESHOLD
high_conf_accuracy = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask]) if np.any(high_conf_mask) else 0
low_conf_mask = confidences < CONFIDENCE_THRESHOLD
low_conf_accuracy = np.mean(y_true[low_conf_mask] == y_pred[low_conf_mask]) if np.any(low_conf_mask) else 0

# Print comprehensive results
print("\n" + "="*60)
print("FINAL TEST RESULTS WITH CONFIDENCE SCORES")
print("="*60)
print(f"Overall Accuracy: {accuracy:.4f} (95% CI: {acc_lower:.4f}-{acc_upper:.4f})")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"\nConfidence Analysis (Threshold = {CONFIDENCE_THRESHOLD}):")
print(f"  - High-confidence predictions: {np.sum(high_conf_mask)}/{len(confidences)} ({np.sum(high_conf_mask)/len(confidences)*100:.1f}%)")
print(f"  - High-confidence accuracy: {high_conf_accuracy:.4f}")
print(f"  - Low-confidence predictions: {np.sum(low_conf_mask)}/{len(confidences)} ({np.sum(low_conf_mask)/len(confidences)*100:.1f}%)")
print(f"  - Low-confidence accuracy: {low_conf_accuracy:.4f}")
print(f"  - Average confidence score: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}")

# Per-class confidence analysis
print("\nPer-Class Performance with Confidence:")
for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    if np.any(class_mask):
        class_acc = np.mean(y_pred[class_mask] == i)
        class_conf = np.mean(confidences[class_mask])
        print(f"  {class_name}: Acc={class_acc:.4f}, Avg Confidence={class_conf:.4f}")
    else:
        print(f"  {class_name}: No samples in test set")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - ConvNeXt Tiny')
plt.savefig(os.path.join(RESULTS, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RESULTS, 'classification_report.csv'))

# ROC Curves
plt.figure(figsize=(10, 8))
for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true==i, y_prob[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})', linewidth=2)

plt.plot([0,1], [0,1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - ConvNeXt Tiny', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Confidence distribution plot
plt.figure(figsize=(10, 6))
correct_conf = confidences[y_true == y_pred]
incorrect_conf = confidences[y_true != y_pred]
if len(correct_conf) > 0 and len(incorrect_conf) > 0:
    plt.hist([correct_conf, incorrect_conf], 
             bins=20, label=['Correct', 'Incorrect'], alpha=0.7, stacked=False)
elif len(correct_conf) > 0:
    plt.hist(correct_conf, bins=20, label=['Correct'], alpha=0.7)
elif len(incorrect_conf) > 0:
    plt.hist(incorrect_conf, bins=20, label=['Incorrect'], alpha=0.7)
plt.xlabel('Confidence Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(RESULTS, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Confidence vs Accuracy plot
thresholds = np.arange(0.5, 1.0, 0.05)
acc_at_threshold = []
coverage_at_threshold = []

for thresh in thresholds:
    high_conf = confidences >= thresh
    if np.any(high_conf):
        acc_at_threshold.append(np.mean(y_true[high_conf] == y_pred[high_conf]))
        coverage_at_threshold.append(np.mean(high_conf))
    else:
        acc_at_threshold.append(0)
        coverage_at_threshold.append(0)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(thresholds, acc_at_threshold, 'b-o', label='Accuracy at threshold', linewidth=2)
ax1.set_xlabel('Confidence Threshold', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(thresholds, coverage_at_threshold, 'r-s', label='Coverage', linewidth=2)
ax2.set_ylabel('Coverage (fraction of data)', fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Accuracy vs Coverage at Different Confidence Thresholds', fontsize=14)
plt.savefig(os.path.join(RESULTS, 'confidence_threshold_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# -----------------------------
# 8) Grad-CAM with Confidence Scores
# -----------------------------
def find_convnext_target_layer(model):
    # Search last stage for Conv2d
    for name, module in model.backbone.stages[-1].named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"[INFO] Found target layer: stages[-1].{name}")
            return module
    # Fallback
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"[INFO] Fallback target layer: {name}")
            return module
    raise ValueError("No Conv2d layer found for Grad-CAM")

target_layer = find_convnext_target_layer(model)
cam = GradCAM(model=model, target_layers=[target_layer])
save_gradcam_dir = os.path.join(RESULTS, "gradcam_all")
os.makedirs(save_gradcam_dir, exist_ok=True)

# Create subdirectories for correct/incorrect predictions
correct_dir = os.path.join(save_gradcam_dir, "correct_predictions")
incorrect_dir = os.path.join(save_gradcam_dir, "incorrect_predictions")
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

gradcam_results = []
confidence_by_class = {cls: {'correct': [], 'confidences': []} for cls in class_names}

print("\n[INFO] Generating Grad-CAM visualizations...")
for idx, (img_path, label) in enumerate(test_dataset.samples):
    img = Image.open(img_path).convert("RGB")
    img_tensor = val_transforms(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_class = pred_class.item()
    
    # Store confidence by class
    is_correct = (pred_class == label)
    confidence_by_class[class_names[label]]['confidences'].append(confidence)
    confidence_by_class[class_names[label]]['correct'].append(1 if is_correct else 0)
    
    # Generate Grad-CAM
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    rgb_img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True, image_weight=0.5, colormap=cv2.COLORMAP_JET)
    
    # Add confidence score to visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(visualization)
    color = 'green' if is_correct else 'red'
    status = "? CORRECT" if is_correct else "? INCORRECT"
    title = f"Pred: {class_names[pred_class]} | True: {class_names[label]}\nConfidence: {confidence:.3f} | {status}"
    plt.title(title, color=color, fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Save in appropriate directory
    fname = f"{idx:04d}_pred_{class_names[pred_class]}_true_{class_names[label]}_conf_{confidence:.3f}.png"
    if is_correct:
        save_path = os.path.join(correct_dir, fname)
    else:
        save_path = os.path.join(incorrect_dir, fname)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    gradcam_results.append({
        'image': os.path.basename(img_path),
        'true_class': class_names[label],
        'pred_class': class_names[pred_class],
        'confidence': confidence,
        'correct': is_correct,
        'visualization': fname
    })
    
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx+1}/{len(test_dataset.samples)} images")

# Save Grad-CAM results to CSV
gradcam_df = pd.DataFrame(gradcam_results)
gradcam_df.to_csv(os.path.join(RESULTS, 'gradcam_predictions_with_confidence.csv'), index=False)

# -----------------------------
# 9) Enhanced HTML Visualization
# -----------------------------
html_file = os.path.join(RESULTS, 'visualize_gradcam.html')
with open(html_file, 'w') as f:
    f.write('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ConvNeXt Tiny - GradCAM Results with Confidence</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; margin-top: 30px; }
            .summary { background: white; padding: 15px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
            .card { 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 10px; 
                width: 320px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .card:hover { transform: translateY(-5px); }
            .card.correct { border-left: 5px solid #4CAF50; }
            .card.incorrect { border-left: 5px solid #f44336; }
            .card img { width: 100%; height: auto; border-radius: 4px; }
            .info { padding: 10px 5px; }
            .confidence { font-weight: bold; font-size: 1.1em; }
            .correct-text { color: #4CAF50; }
            .incorrect-text { color: #f44336; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #2196F3; color: white; }
        </style>
    </head>
    <body>
        <h1>?? ConvNeXt Tiny - Heel Spur Classification</h1>
        <h1>Grad-CAM Visualization with Confidence Scores</h1>
        
        <div class="summary">
            <h2>?? Overall Performance</h2>
            <p><strong>Overall Accuracy:</strong> ''' + f"{accuracy:.4f} (95% CI: {acc_lower:.4f}-{acc_upper:.4f})" + '''</p>
            <p><strong>Balanced Accuracy:</strong> ''' + f"{balanced_acc:.4f}" + '''</p>
            <p><strong>Average Confidence:</strong> ''' + f"{np.mean(confidences):.4f} ± {np.std(confidences):.4f}" + '''</p>
            <p><strong>High-Confidence Accuracy (>{}):</strong> '''.format(CONFIDENCE_THRESHOLD) + f"{high_conf_accuracy:.4f} ({np.sum(high_conf_mask)}/{len(confidences)} samples)" + '''</p>
        </div>
        
        <div class="summary">
            <h2>?? Per-Class Confidence Analysis</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Accuracy</th>
                    <th>Avg Confidence</th>
                    <th>Correct Predictions</th>
                </tr>
    ''')
    
    for cls in class_names:
        cls_data = confidence_by_class[cls]
        if len(cls_data['correct']) > 0:
            cls_acc = np.mean(cls_data['correct'])
            cls_conf = np.mean(cls_data['confidences'])
            f.write(f'''
                <tr>
                    <td>{cls}</td>
                    <td>{cls_acc:.4f}</td>
                    <td>{cls_conf:.4f}</td>
                    <td>{sum(cls_data['correct'])}/{len(cls_data['correct'])}</td>
                </tr>
            ''')
    
    f.write('''
            </table>
        </div>
        
        <h2>? Correct Predictions</h2>
        <div class="container">
    ''')
    
    # Display correct predictions
    correct_count = 0
    for result in gradcam_results:
        if result['correct'] and correct_count < 20:
            card_class = "correct"
            f.write(f'''
            <div class="card {card_class}">
                <img src="gradcam_all/correct_predictions/{result['visualization']}" alt="{result['image']}">
                <div class="info">
                    <p><strong>Image:</strong> {result['image']}</p>
                    <p><strong>Predicted:</strong> {result['pred_class']}</p>
                    <p><strong>True:</strong> {result['true_class']}</p>
                    <p class="confidence correct-text">Confidence: {result['confidence']:.3f}</p>
                </div>
            </div>
            ''')
            correct_count += 1
    
    f.write('''
        </div>
        
        <h2>? Incorrect Predictions</h2>
        <div class="container">
    ''')
    
    # Display incorrect predictions
    for result in gradcam_results:
        if not result['correct']:
            card_class = "incorrect"
            f.write(f'''
            <div class="card {card_class}">
                <img src="gradcam_all/incorrect_predictions/{result['visualization']}" alt="{result['image']}">
                <div class="info">
                    <p><strong>Image:</strong> {result['image']}</p>
                    <p><strong>Predicted:</strong> {result['pred_class']}</p>
                    <p><strong>True:</strong> {result['true_class']}</p>
                    <p class="confidence incorrect-text">Confidence: {result['confidence']:.3f}</p>
                </div>
            </div>
            ''')
    
    f.write('''
        </div>
    </body>
    </html>
    ''')

# -----------------------------
# 10) Final Summary Report
# -----------------------------
summary_report = f"""
================================================================================
CONVNEXT TINY - HEEL SPUR CLASSIFICATION - FINAL REPORT
================================================================================
Date: {pd.Timestamp.now()}
Results Directory: {RESULTS}

DATASET INFORMATION:
- Classes: {class_names}
- Train samples: {len(train_dataset)}
- Validation samples: {len(val_dataset)}
- Test samples: {len(test_dataset)}

MODEL ARCHITECTURE:
- Backbone: ConvNeXt Tiny (pretrained on ImageNet)
- Input size: {IMAGE_SIZE}x{IMAGE_SIZE}
- Loss function: CrossEntropy with class weights
- Optimizer: Adam (LR={LR})
- Scheduler: ReduceLROnPlateau

TRAINING DETAILS:
- Total epochs: {EPOCHS}
- Batch size: {BATCH_SIZE}
- Best validation accuracy: {best_val_acc:.4f}

TEST RESULTS:
- Overall Accuracy: {accuracy:.4f} (95% CI: {acc_lower:.4f}-{acc_upper:.4f})
- Balanced Accuracy: {balanced_acc:.4f}
- Average Confidence: {np.mean(confidences):.4f} ± {np.std(confidences):.4f}

CONFIDENCE ANALYSIS (Threshold={CONFIDENCE_THRESHOLD}):
- High-confidence predictions: {np.sum(high_conf_mask)}/{len(confidences)} ({np.sum(high_conf_mask)/len(confidences)*100:.1f}%)
- High-confidence accuracy: {high_conf_accuracy:.4f}
- Low-confidence predictions: {np.sum(low_conf_mask)}/{len(confidences)} ({np.sum(low_conf_mask)/len(confidences)*100:.1f}%)
- Low-confidence accuracy: {low_conf_accuracy:.4f}

PER-CLASS PERFORMANCE:
"""
for i, cls in enumerate(class_names):
    class_mask = y_true == i
    if np.any(class_mask):
        class_acc = np.mean(y_pred[class_mask] == i)
        class_conf = np.mean(confidences[class_mask])
        summary_report += f"  {cls}: Accuracy={class_acc:.4f}, Avg Confidence={class_conf:.4f}\n"

summary_report += f"""
OUTPUT FILES GENERATED:
1. training_curves.png - Training/validation accuracy and loss curves
2. confusion_matrix.png - Confusion matrix visualization
3. classification_report.csv - Detailed per-class metrics
4. roc_curves.png - ROC curves with AUC scores
5. confidence_distribution.png - Distribution of confidence scores
6. confidence_threshold_analysis.png - Accuracy vs coverage analysis
7. gradcam_predictions_with_confidence.csv - Detailed predictions with confidence
8. visualize_gradcam.html - Interactive HTML report with Grad-CAM visualizations
9. gradcam_all/ - Directory containing all Grad-CAM visualizations
10. best_model.pth - Best performing model weights
11. convnext_model_final.pth - Final model after all epochs

================================================================================
"""

# Save summary report
with open(os.path.join(RESULTS, 'final_summary_report.txt'), 'w') as f:
    f.write(summary_report)

print(summary_report)
print(f"\n[SUCCESS] All results saved in: {RESULTS}")
print(f"[SUCCESS] HTML report: {os.path.join(RESULTS, 'visualize_gradcam.html')}")
print(f"[SUCCESS] Final model: {os.path.join(RESULTS, 'best_model.pth')}")
