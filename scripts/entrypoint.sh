#!/bin/bash
scan_path="$1"
if [ -z "$scan_path" ]; then
    echo "Please provide a scan path, cannot find it or is empty."
    echo "Usage: infer_ensemble.sh <scan_path>"
    exit 1
fi
model_path="/opt/mindGlide/models/_20240404_conjurer_trained_dice_7733.pt"
#model_path1="/opt/mindGlide/models/model_0_net_key_metric=0.7627.pt"
#model_path2="/opt/mindGlide/models/model_2_net_key_metric=0.7541.pt"
#model_path3="/opt/mindGlide/models/model_2_net_key_metric=0.7579.pt"
#model_path4="/opt/mindGlide/models/model_3_net_key_metric=0.7713.pt"
#model_path5="/opt/mindGlide/models/model_3_net_key_metric=0.7717.pt"
#model_path6="/opt/mindGlide/models/model_4_net_key_metric=0.7645.pt"
#model_path7="/opt/mindGlide/models/model_5_net_key_metric=0.7866.pt"
#model_path8="/opt/mindGlide/models/model_6_net_key_metric=0.7723.pt"
#model_path9="/opt/mindGlide/models/model_7_net_key_metric=0.7634.pt"
#model_path10="/opt/mindGlide/models/model_8_net_key_metric=0.7489.pt"
#model_path11="/opt/mindGlide/models/model_9_net_key_metric=0.7738.pt"

export PYTHONPATH="/opt/:/opt/monai:${PYTHONPATH}"
command="python /opt/mindGlide/mindGlide/run_inference.py  --model_file_paths  ${model_path} \
--scan_path ${scan_path}"
#command="python /opt/mindGlide/mindGlide/run_inference.py  --model_file_paths  ${model_path1} ${model_path2} \
#${model_path3} ${model_path4} ${model_path5} ${model_path6} ${model_path7} \
#${model_path8} ${model_path9} ${model_path10} ${model_path11} ${model_path12} \
#${model_path13} \
#--scan_path ${scan_path}"
#command="python /opt/mindGlide/mindGlide/run_inference.py  --model_file_paths  ${model_path1} \
#${model_path2} \
#${model_path13} \
#--scan_path ${scan_path}"
echo $command
$command
