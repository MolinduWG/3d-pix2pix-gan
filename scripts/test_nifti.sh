#!/bin/bash
# Inference script for NIfTI dataset
# Runs inference on test data and saves reconstructed volumes

python3 inference_nifti.py \
  --dataroot /kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset \
  --name mri_super_resolution \
  --model pix2pix3d \
  --dataset_mode nifti \
  --input_nc 1 \
  --output_nc 1 \
  --depthSize 64 \
  --fineSize 64 \
  --which_model_netG unet_128 \
  --which_epoch latest \
  --phase test \
  --results_dir ./results
