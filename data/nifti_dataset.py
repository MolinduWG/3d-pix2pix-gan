import os.path
from data.base_dataset import BaseDataset
from util import nifti_utils
import torch
import numpy as np

class NiftiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        # For Kaggle structure: dataroot/High-Res and dataroot/Low-Res
        # Low-res files have 'lowres_' prefix
        self.dir_high_res = os.path.join(opt.dataroot, 'High-Res')
        self.dir_low_res = os.path.join(opt.dataroot, 'Low-Res')
        
        # Get all high-res files
        self.high_res_paths = self.make_dataset(self.dir_high_res)
        self.high_res_paths = sorted(self.high_res_paths)
        
        # Patch configuration
        self.patch_size = (opt.depthSize, opt.fineSize, opt.fineSize)
        # Default stride to patch size (non-overlapping)
        self.stride = self.patch_size 
        
        # Pre-calculate all patches
        self.patches = []
        
        for path_high_res in self.high_res_paths:
            # Get corresponding low-res file
            filename = os.path.basename(path_high_res)
            low_res_filename = 'lowres_' + filename
            path_low_res = os.path.join(self.dir_low_res, low_res_filename)
            
            # Check if low-res file exists
            if not os.path.exists(path_low_res):
                print(f"Warning: Low-res file not found for {filename}, skipping...")
                continue
            
            # Load header to get shape
            try:
                import nibabel as nib
                img = nib.load(path_high_res)
                shape = img.shape
                
                coords = nifti_utils.get_patch_coords(shape, self.patch_size, self.stride)
                
                for coord in coords:
                    self.patches.append({
                        'high_res_path': path_high_res,
                        'low_res_path': path_low_res,
                        'coord': coord,
                        'shape': shape
                    })
            except Exception as e:
                print(f"Error processing {path_high_res}: {e}")
                continue

    def make_dataset(self, dir):
        images = []
        if not os.path.isdir(dir):
            print(f'Warning: {dir} is not a valid directory')
            return images
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __getitem__(self, index):
        patch_info = self.patches[index]
        
        # Load volumes (this is inefficient if we reload for every patch, but OS cache helps)
        # A is the input (low-res), B is the target (high-res)
        low_res_data, _ = nifti_utils.load_nifti(patch_info['low_res_path'])
        high_res_data, _ = nifti_utils.load_nifti(patch_info['high_res_path'])
        
        low_res_patch = nifti_utils.extract_patch(low_res_data, patch_info['coord'], self.patch_size)
        high_res_patch = nifti_utils.extract_patch(high_res_data, patch_info['coord'], self.patch_size)
        
        # Convert to tensors
        # Add channel dimension [1, D, H, W]
        A_tensor = torch.from_numpy(low_res_patch).float().unsqueeze(0)
        B_tensor = torch.from_numpy(high_res_patch).float().unsqueeze(0)
        
        item = {
            'A': A_tensor,
            'B': B_tensor,
            'A_paths': patch_info['low_res_path'],
            'B_paths': patch_info['high_res_path'],
            'coord': str(patch_info['coord'])
        }
        
        return item

    def __len__(self):
        return len(self.patches)

    def name(self):
        return 'NiftiDataset'
