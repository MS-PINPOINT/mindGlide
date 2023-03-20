# please replace the weight variable into your actual weight
echo "working directory is:"
pwd
model_path1="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold13_baseline/net_key_metric=0.7713.pt"
model_path2="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold5_baseline/net_key_metric=0.7866.pt"
model_path3="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold14_baseline/net_key_metric=0.7645.pt"
model_path4="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold8_baseline/net_key_metric=0.7489.pt"
modal_path5="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold11_baseline/net_key_metric=0.7637.pt"
model_path6="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold6_baseline/net_key_metric=0.7723.pt"
model_path7="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold3_baseline/net_key_metric=0.7717.pt"
model_path8="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold14_baseline/net_key_metric=0.7645.pt"
model_path9="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold7_baseline/net_key_metric=0.7634.pt"
model_path10="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold9_baseline/net_key_metric=0.7738.pt"
model_path11="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold12_baseline/net_key_metric=0.7541.pt"
model_path12="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold2_baseline/net_key_metric=0.7579.pt"
model_path13="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold10_baseline/net_key_metric=0.7627.pt"

scan_path="/mounts/auto/arman7/training_files/inference/_20/ensemble/20160412-bsl_t1c-1.nii.gz"
export PYTHONPATH="/opt/:/opt/monai:${PYTHONPATH}"
command="python run_inference.py  --model_file_paths  ${model_path1} ${model_path2} --scan_path ${scan_path}"
echo $command
$command
