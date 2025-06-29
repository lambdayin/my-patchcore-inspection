datapath=/mnt/lustre/GPU2/home/yinlei/projects/interest/patchcore_projects/patchcore-inspection/datasets/mvtec_dataset
loadpath=/mnt/lustre/GPU2/home/yinlei/projects/interest/patchcore_projects/patchcore-inspection/results/MVTecAD_Results

# modelfolder=IM320_WR50_L2-3_P001_D1024-1024_PS-5_AN-3_S39
modelfolder=IM320_Ensemble_L2-3_P001_D1024-384_PS-5_AN-5_S88
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable')
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

python bin/load_and_evaluate_patchcore.py --gpu 3 --seed 0  $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
