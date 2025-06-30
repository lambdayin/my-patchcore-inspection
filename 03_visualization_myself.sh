#!/bin/bash

datapath=/home/yinlei/projects/interest/patchcore_projects/my-patchcore-inspection/datasets_all/mvtec_dataset
loadpath=/home/yinlei/projects/interest/patchcore_projects/my-patchcore-inspection/results/MVTecAD_Results

modelfolder=IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S39
savefolder=visualization_results'/MVTecAD_Results/'$modelfolder

datasets=('bottle'  'cable')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python -m bin.visualize_patchcore --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
