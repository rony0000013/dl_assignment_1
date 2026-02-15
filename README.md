# CNN Stress-Testing Assignment

This repository contains implementation for CNN stress-testing assignment on CIFAR-10 dataset using ResNet-18 architecture.

## Team Members

- **Student 1**: Rounak Sen
- **Student 2**: Dishant Nibedit
- **Student 3**: Arghya Jana
- **Student 4**: Kaushik Kumar Das  

## Project Overview

This assignment focuses on understanding CNN behavior through systematic experimentation, failure analysis, and explainability techniques rather than just maximizing accuracy.

### Key Components:
- **Dataset**: CIFAR-10 (10 classes, 32x32 RGB images)
- **Architecture**: Custom ResNet-18 implementation
- **Random Seed**: 42 (ensures reproducibility)
- **Framework**: PyTorch
- **Optimizer**: Adam with weight decay (1e-4)

## Files Structure

```
DL/
├── assignment.ipynb          # Main implementation notebook
├── training_curves.png       # Training/validation loss and accuracy curves
├── model_comparison.png      # Baseline vs improved model comparison
├── failure_case_1.png        # Grad-CAM analysis of failure case 1
├── failure_case_2.png        # Grad-CAM analysis of failure case 2
├── failure_case_3.png        # Grad-CAM analysis of failure case 3
├── analysis_report.txt       # Comprehensive analysis report
├── data/                     # CIFAR-10 dataset (auto-downloaded)
└── README.md                 # This file
```

## Requirements

Install the required packages using:

```bash
pip install torch torchvision matplotlib numpy opencv-python
```

Or using conda:

```bash
conda install pytorch torchvision matplotlib numpy opencv -c pytorch
```

## How to Run

1. **Open the notebook**:
   ```bash
   jupyter notebook assignment.ipynb
   ```

2. **Run all cells sequentially**:
   - Cell 1: Imports and setup
   - Cell 2: ResNet-18 architecture definition
   - Cell 3: Data loading and preprocessing
   - Cell 4: Training and evaluation functions
   - Cell 5: Baseline model training (30 epochs)
   - Cell 6: Training curves visualization
   - Cell 7: Test evaluation and failure case discovery
   - Cell 8: Grad-CAM explainability analysis
   - Cell 9: Constrained improvement (enhanced data augmentation)
   - Cell 10: Model comparison
   - Cell 11: Failure analysis comparison
   - Cell 12: Final analysis report generation

3. **Expected Runtime**:
   - Training: ~30-45 minutes on GPU, ~2-3 hours on CPU
   - Total notebook execution: ~1 hour on GPU

## Key Features

### Baseline Model
- Custom ResNet-18 implementation from scratch
- Trained for 30 epochs with Adam optimizer
- Cosine annealing learning rate scheduler
- Fixed random seed (42) for reproducibility

### Failure Case Analysis
- Identifies high-confidence incorrect predictions (>80% confidence)
- Provides detailed analysis of why the model failed
- Includes visual explanations using Grad-CAM with proper heatmap resizing

### Explainability
- Grad-CAM implementation for visualizing model attention
- Heatmaps show which regions influenced decisions
- Overlays help understand model reasoning
- Proper image resizing for accurate localization

### Constrained Improvement
- Single modification: Enhanced data augmentation
- Includes RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing
- Demonstrates impact on robustness and accuracy
- **Result: 5.90% accuracy improvement (85.85% → 91.75%)**

### Generated Outputs
- **training_curves.png**: Loss and accuracy progression
- **model_comparison.png**: Baseline vs improved model performance
- **failure_case_*.png**: Visual analysis of top 3 failure cases
- **analysis_report.txt**: Comprehensive findings and insights

## Experimental Results

### Baseline Model Performance
- **Test Accuracy**: 85.85%
- **High-Confidence Failures**: 891 cases
- **Training Behavior**: Achieved 100% training accuracy (overfitting)

### Improved Model Performance
- **Test Accuracy**: 91.75%
- **High-Confidence Failures**: 404 cases
- **Failure Reduction**: 487 fewer high-confidence failures
- **Better Generalization**: Reduced overfitting through data augmentation

### Key Findings
1. **Data augmentation effectiveness**: 5.90% accuracy improvement
2. **Failure patterns**: Most failures involved visually similar classes (bird↔dog, airplane↔ship)
3. **Model confidence**: High confidence even in incorrect predictions
4. **Attention patterns**: Grad-CAM revealed focus on both relevant and background regions

## Reproducibility

- Fixed random seed (42) ensures consistent results
- All hyperparameters are explicitly defined
- Code is fully self-contained
- No external pretrained weights used
- Adam optimizer with weight decay for better generalization

## Notes

- The CIFAR-10 dataset will be automatically downloaded to the `data/` directory
- GPU is recommended for faster training
- All visualizations are saved as high-resolution PNG files
- The analysis report provides key insights and reflections
- Grad-CAM heatmaps are properly resized to match input image dimensions

## Assignment Requirements Met

✅ Dataset: CIFAR-10 (official train-test split)  
✅ Architecture: ResNet-18 (custom implementation)  
✅ Fixed random seed: 42  
✅ Baseline training with metrics  
✅ 3+ failure cases with analysis  
✅ Explainability techniques (Grad-CAM)  
✅ Single constrained improvement (data augmentation)  
✅ Comprehensive analysis and reflection  
✅ Reproducible codebase  

For questions or issues, refer to the detailed comments within the notebook cells.
