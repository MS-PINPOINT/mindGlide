# please replace the weight variable into your actual weight
echo "working directory is:"
pwd
model_path="/mounts/auto/arman7/training_files/inference/_20/models/runs_12_fold13_baseline/net_key_metric=0.7713.pt"
scan_path="/mounts/auto/arman7/training_files/inference/_20/ensemble/20160412-bsl_t1c-1.nii.gz"
export PYTHONPATH="/opt/:/opt/monai:${PYTHONPATH}"
command="python run_inference.py  --model_file_paths  ${model_path} --scan_path ${scan_path}"
echo $command
$command
