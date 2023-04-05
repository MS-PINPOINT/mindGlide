#!/bin/bash
model_path1="/opt/mindGlide/models/model_0_net_key_metric=0.7627.pt"
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

scan_path="/mounts/auto/arman7/workflows/mindGlide/test/20121229_t2.nii.gz"
export PYTHONPATH="/opt/:/opt/monai:${PYTHONPATH}"
command="python /opt/mindGlide/mindGlide/fine_tuning.py  \
--model_weight ${model_path1}\ 
--scan_path ${scan_path}"
echo $command
$command
