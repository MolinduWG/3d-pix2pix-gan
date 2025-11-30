import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import nifti_utils
import torch
import numpy as np
from tqdm import tqdm

def run_inference():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataset_mode = 'nifti'
    
    # Create data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    
    # Create model
    model = create_model(opt)
    
    # Dictionary to store patches for reconstruction
    # Key: volume_path, Value: list of (patch, coord)
    reconstruction_buffer = {}
    
    print("Starting inference...")
    
    for i, data in enumerate(tqdm(dataset)):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        
        fake_B = visuals['fake_B']
        
        path = data['A_paths'][0]
        coord_str = data['coord'][0]
        coord = eval(coord_str)
        
        if path not in reconstruction_buffer:
            reconstruction_buffer[path] = []
            
        reconstruction_buffer[path].append((fake_B, coord))
        
    print("Reconstructing volumes...")
    
    save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for path, patches_data in reconstruction_buffer.items():
        patches = [p[0] for p in patches_data]
        coords = [p[1] for p in patches_data]
        
        _, affine = nifti_utils.load_nifti(path)
        
        import nibabel as nib
        img = nib.load(path)
        shape = img.shape
        
        patch_size = patches[0].shape
        
        reconstructed_volume = nifti_utils.stitch_volume(patches, coords, shape, patch_size)
        
        filename = os.path.basename(path)
        save_path = os.path.join(save_dir, filename)
        nifti_utils.save_nifti(reconstructed_volume, save_path, affine)
        
        print(f"Saved {save_path}")

if __name__ == '__main__':
    run_inference()
