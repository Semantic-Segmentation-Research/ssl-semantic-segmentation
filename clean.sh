#! /bin/bash

backups_dir="/home/dev/experiments/backups"
codes_dir="/home/dev/experiments/codes"
logs_dir="/home/dev/experiments/logs"
models_dir="/home/dev/experiments/models"

model_name="v1.3.0_custom_resnet50_xca"
rm -rf ${backups_dir}/${model_name}
rm -rf ${codes_dir}/${model_name}
rm -rf ${logs_dir}/${model_name}
rm -rf ${models_dir}/${model_name}

# for d in "${backups_dir}/${model_name}" "${codes_dir}/${model_name}" "${logs_dir}/${model_name}" "${models_dir}/${model_name}"; do
