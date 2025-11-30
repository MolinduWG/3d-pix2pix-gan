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
        self.dir_A = os.path.join(opt.dataroot, 'High-Res')
        self.dir_B = os.path.join(opt.dataroot, 'Low-Res')
        
        self.patch_size = (opt.depthSize, opt.fineSize, opt.fineSize)
        self.stride = self.patch_size 
        
        all_high_res_files = sorted(self.make_dataset(self.dir_A))
        
        random.seed(42)
        random.shuffle(all_high_res_files)
        
        split_idx = int(len(all_high_res_files) * 0.9)
        if opt.phase == 'train':
            self.A_paths = all_high_res_files[:split_idx]
        elif opt.phase == 'val':
            self.A_paths = all_high_res_files[split_idx:]
        else:
            self.A_paths = all_high_res_files
            
        self.B_paths = []
        for path in self.A_paths:
            filename = os.path.basename(path)
            low_res_filename = 'lowres_' + filename
            low_res_path = os.path.join(self.dir_B, low_res_filename)
            if os.path.exists(low_res_path):
                self.B_paths.append(low_res_path)
            else:
                print(f"Warning: Corresponding low-res file not found for {filename}")
        
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.patches = []
        
        for i, path_high_res in enumerate(self.A_paths):
            path_low_res = self.B_paths[i]
            
            if not os.path.exists(path_low_res):
                print(f"Warning: Low-res file not found for {os.path.basename(path_high_res)}, skipping...")
                continue
            
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
        
        low_res_data, _ = nifti_utils.load_nifti(patch_info['low_res_path'])
        high_res_data, _ = nifti_utils.load_nifti(patch_info['high_res_path'])
        
        low_res_patch = nifti_utils.extract_patch(low_res_data, patch_info['coord'], self.patch_size)
        high_res_patch = nifti_utils.extract_patch(high_res_data, patch_info['coord'], self.patch_size)
        
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
