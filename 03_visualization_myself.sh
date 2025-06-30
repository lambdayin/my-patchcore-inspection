#!/bin/bash

datapath=./mvtec_anomaly_detection
loadpath=./patchcore_model_weights

modelfolder=IM224_WR101_L2-3_P001_D1024-384_PS-3_AN-1_RN101_L2-3_P001_D1024-384_PS-3_AN-1_DN201_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=visualization_results/$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python -m bin.visualize_patchcore --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
