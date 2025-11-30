#!/bin/bash
# Validation script for Kaggle NIfTI dataset
# Runs validation on the 10% held-out data

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
  --gpu_ids 0 \
  --which_model_netG unet_128 \
  --display_id 0 \
  --no_html \
  --phase val \
  --which_epoch latest \
  --continue_train
