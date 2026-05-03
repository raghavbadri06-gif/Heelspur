 ZHaCTreS-Net-JR: Attention-Enhanced ConvNeXt with PSO Optimization for Heel Spur Classification

1. Overview

This repository implements ZHaCTreS-Net-JR for automated heel spur classification from foot X-ray images.

Classes:
- heelspur
- normal
- sever

The framework integrates ConvNeXt-Tiny backbone, CBAM attention, PSO hyperparameter optimization, multi-stage feature extraction, 
and OCR-based clinical metadata extraction.

---

2. Environment

Tested configuration:

- Python 3.12+
- PyTorch 2.5+
- CUDA-enabled GPU (recommended)

Install dependencies:

```bash
pip install torch torchvision timm numpy pandas matplotlib seaborn scikit-learn scipy opencv-python pytesseract pyswarm pytorch-grad-cam pillow, these are done
till the branch contribution extraction process.
For the statistical analysis matlab was used in this study
#dataset
Dataset: https://www.kaggle.com/datasets/osamahtaher/heel-dataset
Normal: 1,842 images Heel Spur: 1,316 images Sever (Heel Spur Complications): 798 images

4. Dataset Structure
Organise the data exactly as follows:

text
Traininghs/
├── heelspur/
│   ├── image1.jpg
│   └── ...
├── normal/
│   ├── image1.jpg
│   └── ...
└── sever/
│   ├── image1.jpg
│   └── ...
The code automatically performs stratified split: 70% train, 15% validation, 15% test

6. Running the Experiments
Execute:

bash
python filename.py

This will automatically:

Run PSO hyperparameter optimization (5 parameters, 20 iterations)

Train the full model with optimized hyperparameters (25 epochs)

Perform test evaluation

Generate Grad-CAM visualizations

Compute branch contribution analysis

Extract OCR metadata from images

Save all results and metrics

No additional scripts are required.

7. Outputs
All outputs are saved in the specified RESULTS directory:

text
RESULTS/
├── classification_report.csv
├── confusion_matrix.png
├── roc_curves.png
├── accuracy.png
├── loss.png
├── pso_optimization_history.csv
├── optimal_hyperparameters_pso.csv
├── model_final.pth
├── gradcam_visualizations/
│   └── *_gradcam.png
├── branch_contributions.csv
├── stage_contributions/
│   ├── stage_analysis.csv
│   ├── stage_statistics.csv
│   └── stage_weights.png
├── overall_metrics.csv
└── computation_costs.csv







