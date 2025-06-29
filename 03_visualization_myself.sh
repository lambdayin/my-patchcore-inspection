#!/bin/bash

# =============================================================================
# PatchCore Visualization Script
# This script generates visualization results for PatchCore anomaly detection
# =============================================================================

# Configuration paths - MODIFY THESE ACCORDING TO YOUR SETUP
datapath=/mnt/lustre/GPU2/home/yinlei/projects/interest/patchcore_projects/patchcore-inspection/datasets/mvtec_dataset
loadpath=/mnt/lustre/GPU2/home/yinlei/projects/interest/patchcore_projects/patchcore-inspection/results/MVTecAD_Results

# Model configuration - Choose the model folder you want to visualize
# Uncomment the model configuration you want to use:

# For ensemble models (IM320):
# modelfolder=IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S39
modelfolder=IM320_DENSENET201_L2-3_P001_D1024-1024_PS-5_AN-3_S39
# modelfolder=IM320_Ensemble_L2-3_P001_D1024-384_PS-5_AN-5_S88

# For baseline models (IM224):
# modelfolder=IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0

# For ensemble models (IM224):
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1

# Output directory
savefolder=visualization_results'/MVTecAD_Results/'$modelfolder

# Dataset configuration
datasets=('bottle'  'cable')

# Build model flags - one model path per dataset
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))

# Build dataset flags
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

# Image processing parameters
# Adjust these based on your model training configuration:
if [[ $modelfolder == *"IM320"* ]]; then
    # For IM320 models
    resize_param="--resize 366"
    imagesize_param="--imagesize 320"
elif [[ $modelfolder == *"IM224"* ]]; then
    # For IM224 models  
    resize_param="--resize 256"
    imagesize_param="--imagesize 224"
else
    # Default parameters
    resize_param="--resize 366"
    imagesize_param="--imagesize 320"
fi

# GPU configuration
gpu_id=0
seed=0

# =============================================================================
# THRESHOLD CONFIGURATION - CUSTOMIZE ANOMALY DETECTION SENSITIVITY
# =============================================================================

# Threshold mode options:
# - "adaptive"      : Automatic threshold (mean + 2*std) [DEFAULT]
# - "fixed"         : Use fixed threshold value
# - "percentile"    : Use percentile-based threshold (0-100)
# - "std_multiplier": Use mean + threshold_value * std

# Choose threshold mode (uncomment one):
threshold_mode="adaptive"           # Default: automatic threshold
# threshold_mode="fixed"            # Use fixed value
# threshold_mode="percentile"       # Use percentile
# threshold_mode="std_multiplier"   # Use standard deviation multiplier

# Set threshold value based on mode:
# For adaptive mode: leave empty (automatic)
# For fixed mode: absolute threshold value (e.g., 0.5)
# For percentile mode: percentile value 0-100 (e.g., 95)
# For std_multiplier mode: multiplier value (e.g., 2.5)

threshold_value=""  # Leave empty for adaptive mode

# Examples of threshold configurations:
# threshold_mode="fixed" && threshold_value="0.3"           # Fixed threshold at 0.3
# threshold_mode="percentile" && threshold_value="95"       # Use 95th percentile
# threshold_mode="std_multiplier" && threshold_value="2.5"  # mean + 2.5*std

# Build threshold parameters
if [ -n "$threshold_value" ]; then
    threshold_params="--threshold $threshold_value --threshold-mode $threshold_mode"
else
    threshold_params="--threshold-mode $threshold_mode"
fi

# Display configuration
echo "=============================================="
echo "PatchCore Visualization Configuration"
echo "=============================================="
echo "Data path: $datapath"
echo "Model path: $loadpath"
echo "Model folder: $modelfolder"
echo "Output folder: $savefolder"
echo "Datasets: ${datasets[*]}"
echo "Image processing: $resize_param $imagesize_param"
echo "GPU ID: $gpu_id"
echo "Threshold mode: $threshold_mode"
if [ -n "$threshold_value" ]; then
    echo "Threshold value: $threshold_value"
else
    echo "Threshold value: automatic"
fi
echo "=============================================="

# Check if paths exist
if [ ! -d "$datapath" ]; then
    echo "ERROR: Data path does not exist: $datapath"
    echo "Please modify the 'datapath' variable in this script."
    exit 1
fi

if [ ! -d "$loadpath/$modelfolder" ]; then
    echo "ERROR: Model path does not exist: $loadpath/$modelfolder"
    echo "Please modify the 'loadpath' and 'modelfolder' variables in this script."
    exit 1
fi

# Create output directory
mkdir -p "$savefolder"

# Run visualization
echo "Starting PatchCore visualization..."
echo "Command:"
echo "env PYTHONPATH=src python bin/visualize_patchcore.py --gpu $gpu_id --seed $seed $threshold_params $savefolder \\"
echo "patch_core_loader \"${model_flags[@]}\" --faiss_on_gpu \\"
echo "dataset $resize_param $imagesize_param \"${dataset_flags[@]}\" mvtec $datapath"
echo ""

env PYTHONPATH=src python bin/visualize_patchcore.py --gpu $gpu_id --seed $seed $threshold_params $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset $resize_param $imagesize_param "${dataset_flags[@]}" mvtec $datapath

# Check if visualization completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Visualization completed successfully!"
    echo "=============================================="
    echo "Results saved to: $savefolder"
    echo ""
    echo "Output structure:"
    echo "$savefolder/"
    echo "├── bottle/"
    echo "│   ├── [image_name]_mask.png          # Binary anomaly masks"
    echo "│   ├── [image_name]_visualization.png # Overlay visualizations"
    echo "│   └── [image_name]_heatmap.png       # Anomaly heatmaps"
    echo "├── cable/"
    echo "│   └── ..."
    echo "└── ..."
    echo ""
    echo "File types:"
    echo "- *_mask.png: Binary black/white anomaly masks"
    echo "- *_visualization.png: Original image with binary mask overlay (red)"
    echo "- *_heatmap.png: Colored anomaly heatmaps (for reference)"
    echo "=============================================="
else
    echo ""
    echo "ERROR: Visualization failed!"
    echo "Please check the error messages above and verify your configuration."
    exit 1
fi