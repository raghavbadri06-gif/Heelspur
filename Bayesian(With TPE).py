#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
import time
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import timm

# Optimization backend
import optuna                 # Bayesian / TPE
from optuna.samplers import TPESampler

# -----------------------------
# Config
# -----------------------------
ROOT = r"/nfsshare/users/raghavan/heelspurfinal/Traininghs/"
OUT_SPLIT = r"/nfsshare/users/raghavan/heelspurfinal/splitgradcam/"
RESULTS = r"/nfsshare/users/raghavan/heelspurfinal/optunavaidhyal"
os.makedirs(OUT_SPLIT, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25                 # epochs for the FINAL training run
HPO_EPOCHS = 5               # epochs used inside every HPO evaluation (kept equal across methods)
SEED = 42
NUM_WORKERS = 4
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# Fair evaluation budget shared across HPO methods (Bayesian/TPE here,
# PSO's own documented run elsewhere -- see DOCUMENTED_PSO_RESULT below)
N_EVALS = 210

# -----------------------------------------------------------------------
# PSO has already been run and documented separately (population size,
# iterations, inertia/acceleration coefficients, bounds, seed, etc. are in
# the paper/report). Fill in its already-obtained results here so this
# script can compare against them without re-running PSO. Replace the
# placeholder numbers below with your actual documented PSO output.
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# PSO has already been run and documented separately (population size,
# iterations, inertia/acceleration coefficients, bounds, seed, etc. are in
# the paper/report). This block loads its already-obtained results so this
# script can compare against them WITHOUT re-running PSO.
#
# Auto-load path: if your earlier PSO run (the original combined script)
# already wrote out 'optimal_hyperparameters_pso.csv' and
# 'pso_optimization_history.csv' into a results folder, point
# PSO_RESULTS_DIR at that folder and the values below will be filled in
# automatically. If that folder/files aren't found, edit the manual
# fallback values by hand instead.
# -----------------------------------------------------------------------
PSO_RESULTS_DIR = r"/nfsshare/users/raghavan/heelspurfinal/novelswarmpart3_ocrreadcorrectedthada5_final"

def load_documented_pso_result(results_dir):
    """
    Reads optimal_hyperparameters_pso.csv (Parameter/Value rows) and
    pso_optimization_history.csv (one row per PSO evaluation, needs
    columns: lr, dropout_rate, weight_decay, focal_gamma, lambda_ce,
    accuracy) that the ORIGINAL PSO script wrote out. Returns a dict in
    the same shape as DOCUMENTED_PSO_RESULT, or None if the files aren't
    found / can't be parsed.
    """
    hp_path = os.path.join(results_dir, 'optimal_hyperparameters_pso.csv')
    hist_path = os.path.join(results_dir, 'pso_optimization_history.csv')
    if not os.path.exists(hp_path):
        return None
    try:
        hp_df = pd.read_csv(hp_path).set_index('Parameter')['Value']
        result = {
            'method': 'PSO (documented separately)',
            'best_val_acc': float(hp_df['Best Validation Accuracy']),
            'total_time_sec': float(hp_df['PSO Time (seconds)']),
            'n_evaluations': None,  # filled below if history file is present
            'best_params': {
                'lr': float(hp_df['Learning Rate']),
                'dropout_rate': float(hp_df['Dropout Rate']),
                'weight_decay': float(hp_df['Weight Decay']),
                'focal_gamma': float(hp_df['Focal Loss Gamma']),
                'lambda_ce': float(hp_df['Lambda CE']),
            }
        }
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            result['n_evaluations'] = len(hist_df)  # one row per fitness evaluation
        else:
            result['n_evaluations'] = N_EVALS  # fallback assumption
        return result
    except Exception as e:
        print(f"[WARN] Could not auto-load documented PSO results from {results_dir}: {e}")
        return None

DOCUMENTED_PSO_RESULT = load_documented_pso_result(PSO_RESULTS_DIR)

if DOCUMENTED_PSO_RESULT is None:
    print(f"[WARN] No PSO results found at {PSO_RESULTS_DIR}. "
          f"Falling back to the manual DOCUMENTED_PSO_RESULT values below -- "
          f"edit these by hand if the auto-load path is wrong.")
    DOCUMENTED_PSO_RESULT = {
        'method': 'PSO (documented separately)',
        'best_val_acc': None,        # TODO: fill in PSO's reported best validation accuracy
        'n_evaluations': N_EVALS,    # TODO: fill in PSO's actual swarmsize * maxiter
        'total_time_sec': None,      # TODO: fill in PSO's reported optimization time (e.g. 22222)
        'best_params': {
            'lr': None,              # TODO
            'dropout_rate': None,    # TODO
            'weight_decay': None,    # TODO
            'focal_gamma': None,     # TODO
            'lambda_ce': None,       # TODO
        }
    }
else:
    print(f"[INFO] Auto-loaded documented PSO results from {PSO_RESULTS_DIR}")
    print(f"       best_val_acc={DOCUMENTED_PSO_RESULT['best_val_acc']:.4f}, "
          f"n_evaluations={DOCUMENTED_PSO_RESULT['n_evaluations']}, "
          f"total_time_sec={DOCUMENTED_PSO_RESULT['total_time_sec']:.1f}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
# 5) Hyperparameter search space + shared objective function
# =====================================================================
# Order: [lr, dropout_rate, weight_decay, focal_gamma, lambda_ce]
PARAM_NAMES = ['lr', 'dropout_rate', 'weight_decay', 'focal_gamma', 'lambda_ce']
lb = [1e-5, 0.1, 1e-6, 0.5, 0.3]
ub = [1e-3, 0.5, 1e-3, 3.0, 0.9]
BOUNDS = dict(zip(PARAM_NAMES, zip(lb, ub)))

hpo_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, drop_last=True)
hpo_val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# test_loader is defined here (not just in the final-training section) because
# quick_train_test_score() below, used during the val-vs-test gap check, needs
# it before the final training loop runs.
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

# Global log shared by every optimizer so we can build a fair convergence comparison
hpo_log = []  # rows: {method, eval_idx, lr, dropout_rate, weight_decay, focal_gamma, lambda_ce, val_acc, eval_time}

def train_evaluate(lr, dropout_rate, weight_decay, focal_gamma, lambda_ce, method_name, eval_idx):
    """
    Trains a short (HPO_EPOCHS) model with the given hyperparameters and
    returns the best validation accuracy achieved. Every optimizer below
    calls this exact same function, so the comparison is apples-to-apples.
    """
    t0 = time.time()
    print(f"[{method_name} | eval {eval_idx+1}/{N_EVALS}] lr={lr:.6f} dropout={dropout_rate:.3f} "
          f"wd={weight_decay:.6f} gamma={focal_gamma:.2f} lambda_ce={lambda_ce:.2f}")

    model = ConvNeXtAttentionModel(num_classes=num_classes, dropout_rate=dropout_rate).to(DEVICE)
    criterion = CombinedLoss(weight=cls_w, gamma=focal_gamma, lambda_ce=lambda_ce)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    for epoch in range(1, HPO_EPOCHS + 1):
        model.train()
        for imgs, labels in hpo_train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in hpo_val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(hpo_val_loader.dataset)
        val_acc /= len(hpo_val_loader.dataset)
        scheduler.step(val_loss)
        best_val_acc = max(best_val_acc, val_acc)

    eval_time = time.time() - t0
    hpo_log.append({
        'method': method_name, 'eval_idx': eval_idx,
        'lr': lr, 'dropout_rate': dropout_rate, 'weight_decay': weight_decay,
        'focal_gamma': focal_gamma, 'lambda_ce': lambda_ce,
        'val_acc': best_val_acc, 'eval_time_sec': eval_time
    })
    print(f"  -> val_acc={best_val_acc:.4f} ({eval_time:.1f}s)")
    return best_val_acc

# =====================================================================
# 6) Bayesian Optimization / TPE (Optuna, full 5-dim space)
#    Same evaluation budget PSO used: n_trials = N_EVALS
# =====================================================================
def run_bayesian_tpe(n_evals=N_EVALS, seed=SEED):
    method = 'Bayesian_TPE'
    print("\n" + "="*70)
    print(f"{method} (Optuna TPESampler, full 5-dim space, {n_evals} trials)")
    print("="*70)

    counter = {'i': 0}
    t_start = time.time()

    def objective(trial):
        lr = trial.suggest_float('lr', lb[0], ub[0], log=True)
        dropout_rate = trial.suggest_float('dropout_rate', lb[1], ub[1])
        weight_decay = trial.suggest_float('weight_decay', lb[2], ub[2], log=True)
        focal_gamma = trial.suggest_float('focal_gamma', lb[3], ub[3])
        lambda_ce = trial.suggest_float('lambda_ce', lb[4], ub[4])
        acc = train_evaluate(lr, dropout_rate, weight_decay, focal_gamma, lambda_ce, method, counter['i'])
        counter['i'] += 1
        return acc

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_evals)

    elapsed = time.time() - t_start
    best_params = study.best_params
    best_acc = study.best_value
    return best_params, best_acc, elapsed, n_evals

# =====================================================================
# 7) Run Bayesian/TPE in code, and compare against the already-documented
#    PSO result (see DOCUMENTED_PSO_RESULT at the top of this file).
#    PSO itself is NOT re-run here -- it has already been executed and
#    reported separately, including its swarm size, iterations, inertia /
#    acceleration coefficients, bounds, seed, and number of fitness
#    evaluations.
# =====================================================================
print("\n" + "#"*70)
print(f"# HYPERPARAMETER OPTIMIZATION COMPARISON (budget = {N_EVALS} evaluations each)")
print("#"*70)

missing = [k for k, v in DOCUMENTED_PSO_RESULT['best_params'].items() if v is None]
if DOCUMENTED_PSO_RESULT['best_val_acc'] is None or missing:
    raise ValueError(
        "DOCUMENTED_PSO_RESULT is not filled in. Edit the placeholders near the top "
        "of this script (best_val_acc, total_time_sec, n_evaluations, and best_params: "
        f"{missing or list(DOCUMENTED_PSO_RESULT['best_params'].keys())}) with your "
        "already-documented PSO results before running this script."
    )

comparison_rows = []

tpe_params, tpe_acc, tpe_time, tpe_n = run_bayesian_tpe()
comparison_rows.append({'method': 'Bayesian / TPE', 'best_val_acc': tpe_acc,
                         'n_evaluations': tpe_n, 'total_time_sec': tpe_time, 'best_params': tpe_params})

comparison_rows.append({
    'method': DOCUMENTED_PSO_RESULT['method'],
    'best_val_acc': DOCUMENTED_PSO_RESULT['best_val_acc'],
    'n_evaluations': DOCUMENTED_PSO_RESULT['n_evaluations'],
    'total_time_sec': DOCUMENTED_PSO_RESULT['total_time_sec'],
    'best_params': DOCUMENTED_PSO_RESULT['best_params']
})

# Save the raw per-evaluation log (Bayesian/TPE only -- PSO's own
# per-evaluation log lives with its separate documentation)
hpo_log_df = pd.DataFrame(hpo_log)
hpo_log_df.to_csv(os.path.join(RESULTS, 'hpo_full_evaluation_log.csv'), index=False)

# -----------------------------
# 7b) Val-vs-test gap check per method (single split, no extra HPO queries)
# -----------------------------
# This does NOT re-run HPO or touch the val set again. It trains one short
# model per method's already-selected config and scores it on the held-out
# test set, purely to see whether a method's reported val accuracy is
# disproportionately higher than its test accuracy -- a symptom of having
# adaptively queried the validation set, per the reviewer's concern. This
# adds one short training run per method (HPO_EPOCHS each: Bayesian/TPE's
# winner, Bayesian/TPE's winner, and PSO's documented winner), not new HPO
# evaluations,
# and does not change the primary optimization protocol or its results.
print("\n" + "="*70)
print("VAL-VS-TEST GAP CHECK (per method's selected config, single split)")
print("="*70)

def quick_train_test_score(params, epochs=HPO_EPOCHS):
    m = ConvNeXtAttentionModel(num_classes=num_classes, dropout_rate=params['dropout_rate']).to(DEVICE)
    crit = CombinedLoss(weight=cls_w, gamma=params['focal_gamma'], lambda_ce=params['lambda_ce'])
    opt = torch.optim.Adam(m.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    for _ in range(epochs):
        m.train()
        for imgs, labels in hpo_train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = crit(m(imgs), labels)
            loss.backward()
            opt.step()
    m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            pred = m(imgs).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total

for r in comparison_rows:
    test_acc_check = quick_train_test_score(r['best_params'])
    r['test_acc_at_hpo_epochs'] = test_acc_check
    r['val_test_gap'] = r['best_val_acc'] - test_acc_check
    print(f"  {r['method']:22s} val_acc={r['best_val_acc']:.4f}  "
          f"test_acc={test_acc_check:.4f}  gap={r['val_test_gap']:+.4f}")

# Save the summary comparison table
comparison_df = pd.DataFrame([{
    'method': r['method'],
    'best_val_acc': r['best_val_acc'],
    'test_acc_at_hpo_epochs': r['test_acc_at_hpo_epochs'],
    'val_test_gap': r['val_test_gap'],
    'n_evaluations': r['n_evaluations'],
    'total_time_sec': r['total_time_sec'],
    **{f'best_{k}': v for k, v in r['best_params'].items()}
} for r in comparison_rows])
comparison_df.to_csv(os.path.join(RESULTS, 'hpo_method_comparison.csv'), index=False)
print("\n" + "="*70)
print("HPO METHOD COMPARISON SUMMARY")
print("="*70)
print(comparison_df[['method', 'best_val_acc', 'test_acc_at_hpo_epochs', 'val_test_gap',
                      'n_evaluations', 'total_time_sec']].to_string(index=False))
print("\nNote: a larger positive val_test_gap suggests that method's reported validation")
print("accuracy is more optimistic relative to generalization -- i.e. more sensitive to")
print("adaptive querying of the single validation set used across all HPO methods.")

# Bar chart: best validation accuracy vs. test accuracy per method
plot_df = comparison_df.melt(
    id_vars='method', value_vars=['best_val_acc', 'test_acc_at_hpo_epochs'],
    var_name='split', value_name='accuracy'
)
plot_df['split'] = plot_df['split'].map({
    'best_val_acc': 'Validation', 'test_acc_at_hpo_epochs': 'Test'
})
plt.figure(figsize=(8, 5))
sns.barplot(data=plot_df, x='method', y='accuracy', hue='split')
plt.ylabel('Accuracy')
plt.title(f'HPO Method Comparison: Val vs. Test (budget = {N_EVALS} evaluations each)')
plt.xticks(rotation=15)
plt.legend(title='')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'hpo_comparison_val_vs_test_accuracy.png'))
plt.close()

# Convergence plot: best-so-far accuracy vs evaluation index for Random
# Search (the only method actually run here). PSO's own convergence trace
# is part of its separate documentation and is not duplicated in this file.
plt.figure(figsize=(8, 6))
for method in hpo_log_df['method'].unique():
    sub = hpo_log_df[hpo_log_df['method'] == method].sort_values('eval_idx')
    best_so_far = sub['val_acc'].cummax()
    plt.plot(sub['eval_idx'].values + 1, best_so_far.values, marker='o', label=method)
plt.axhline(DOCUMENTED_PSO_RESULT['best_val_acc'], color='gray', linestyle='--',
            label=f"{DOCUMENTED_PSO_RESULT['method']} best (documented)")
plt.xlabel('Evaluation #')
plt.ylabel('Best Validation Accuracy So Far')
plt.title('Bayesian/TPE Convergence vs. Documented PSO Result')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'hpo_convergence_comparison.png'))
plt.close()

# -----------------------------
# Pick the overall winner to drive the final full training run
# -----------------------------
winner_row = max(comparison_rows, key=lambda r: r['best_val_acc'])
print(f"\n[INFO] Overall best method: {winner_row['method']} "
      f"(val_acc={winner_row['best_val_acc']:.4f})")

LR = winner_row['best_params']['lr']
DROPOUT_RATE = winner_row['best_params']['dropout_rate']
WEIGHT_DECAY = winner_row['best_params']['weight_decay']
FOCAL_GAMMA = winner_row['best_params']['focal_gamma']
LAMBDA_CE = winner_row['best_params']['lambda_ce']

final_hyperparams_df = pd.DataFrame({
    'Parameter': ['Winning Method', 'Learning Rate', 'Dropout Rate', 'Weight Decay',
                  'Focal Loss Gamma', 'Lambda CE', 'Best HPO Validation Accuracy'],
    'Value': [winner_row['method'], LR, DROPOUT_RATE, WEIGHT_DECAY, FOCAL_GAMMA,
              LAMBDA_CE, winner_row['best_val_acc']]
})
final_hyperparams_df.to_csv(os.path.join(RESULTS, 'final_selected_hyperparameters.csv'), index=False)
print("[INFO] Final selected hyperparameters saved.")

# -----------------------------
# 8) Final Training Loop with Selected Hyperparameters
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
# test_loader was already created earlier (needed by quick_train_test_score
# during the val-vs-test gap check) -- reused here, not redefined.

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

    print(f"[Epoch {epoch}/{EPOCHS}] Train: {epoch_loss:.4f}, {epoch_acc:.4f} | "
          f"Val: {val_loss:.4f}, {val_acc:.4f} | Time: {epoch_time:.2f}s")
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
# 9) Plots: Accuracy & Loss
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
# 10) Test Evaluation: Confusion Matrix + ROC + Classification Report
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

test_accuracy = (y_true == y_pred).mean() * 100

# -----------------------------
# 11) Save Final Computation / Overall Metrics
# -----------------------------
total_time = time.time() - start_time

computation_df = pd.DataFrame(computation_costs)
computation_df.to_csv(os.path.join(RESULTS, 'computation_costs.csv'), index=False)

overall_metrics = {
    'total_runtime_seconds': total_time,
    'total_runtime_hours': total_time / 3600,
    'bayesian_tpe_time_seconds': tpe_time,
    'documented_pso_time_seconds': DOCUMENTED_PSO_RESULT['total_time_sec'],
    'hpo_winning_method': winner_row['method'],
    'hpo_winning_val_acc': winner_row['best_val_acc'],
    'total_training_time_seconds': total_training_time,
    'average_epoch_time_seconds': total_training_time / EPOCHS,
    'total_epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'total_batches_training': len(train_loader),
    'test_samples': len(test_dataset),
    'test_accuracy_percent': test_accuracy
}
overall_metrics_df = pd.DataFrame([overall_metrics])
overall_metrics_df.to_csv(os.path.join(RESULTS, 'overall_metrics.csv'), index=False)

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(comparison_df[['method', 'best_val_acc', 'test_acc_at_hpo_epochs', 'val_test_gap',
                      'n_evaluations', 'total_time_sec']].to_string(index=False))
print(f"\nWinning HPO method: {winner_row['method']} (val_acc={winner_row['best_val_acc']:.4f})")
print(f"Final Test Accuracy: {test_accuracy:.2f}% ({(y_true == y_pred).sum()}/{len(y_true)})")
print(f"Total Runtime: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
print(f"Training Time (final run): {total_training_time:.2f} seconds")
print(f"Average Epoch Time: {total_training_time/EPOCHS:.2f} seconds")
print("\nAll results saved to:", RESULTS)
print("Done.")
