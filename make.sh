#!/usr/bin/env bash
cd ./utils/
rm nms/cpu_nms.so
rm nms/gpu_nms.so

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace

cd ..
