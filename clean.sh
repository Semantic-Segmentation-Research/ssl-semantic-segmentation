#! /bin/bash

codes_dir="/workspace/experiments/codes"
logs_dir="/workspace/experiments/logs"
models_dir="/workspace/experiments/models"

model_name="v1.6.2_LTU"
rm -rf ${backups_dir}/${model_name}
rm -rf ${codes_dir}/${model_name}
rm -rf ${logs_dir}/${model_name}
rm -rf ${models_dir}/${model_name}

echo "------ $model_name deleted ------"
# for d in "${backups_dir}/${model_name}" "${codes_dir}/${model_name}" "${logs_dir}/${model_name}" "${models_dir}/${model_name}"; do
