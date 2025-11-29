#!/bin/bash
# Example training script for Kaggle NIfTI dataset
# This assumes you're running on Kaggle with the dataset mounted at /kaggle/input/

python3 train.py \
  --dataroot /kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset \
  --name mri_super_resolution \
  --model pix2pix3d \
  --dataset_mode nifti \
  --input_nc 1 \
  --output_nc 1 \
  --depthSize 64 \
  --fineSize 64 \
  --batchSize 1 \
  --niter 100 \
  --niter_decay 100 \
  --gpu_ids 0 \
  --display_freq 10 \
  --print_freq 10 \
  --save_epoch_freq 10 \
  --checkpoints_dir ./checkpoints \
  --results_dir ./results
