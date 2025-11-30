import time
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np

def run_dummy_training():
    # Create dummy options
    class DummyOptions:
        def __init__(self):
            self.dataroot = './dummy_data'
            self.name = 'dummy_test'
            self.model = 'pix2pix3d'
            self.dataset_mode = 'nifti'
            self.input_nc = 1
            self.output_nc = 1
            self.depthSize = 64
            self.fineSize = 64
            self.batchSize = 1
            self.niter = 1
            self.niter_decay = 0
            self.gpu_ids = [] # Use CPU for dummy test
            self.display_freq = 1
            self.print_freq = 1
            self.save_epoch_freq = 1
            self.checkpoints_dir = './checkpoints'
            self.results_dir = './results'
            self.which_model_netG = 'unet_128'
            self.which_model_netD = 'basic'
            self.n_layers_D = 3
            self.norm = 'instance'
            self.use_dropout = False
            self.init_type = 'normal'
            self.mask_type = 'center'
            self.lambda_A = 10.0
            self.pool_size = 0
            self.lr = 0.0002
            self.beta1 = 0.5
            self.isTrain = True
            self.no_html = True # Disable HTML for dummy test
            self.display_id = 0 # Disable Visdom
            self.display_winsize = 256
            self.display_single_pane_ncols = 0
            self.no_lsgan = False
            self.phase = 'train'
            self.which_epoch = 'latest'
            self.continue_train = False
            self.serial_batches = False
            self.nThreads = 0
            self.max_dataset_size = float("inf")
            self.resize_or_crop = 'resize_and_crop'
            self.no_flip = True
            
    opt = DummyOptions()
    
    # Create dummy data
    if not os.path.exists(opt.dataroot):
        os.makedirs(os.path.join(opt.dataroot, 'High-Res'))
        os.makedirs(os.path.join(opt.dataroot, 'Low-Res'))
        
    # Create dummy NIfTI files
    import nibabel as nib
    data = np.random.rand(64, 64, 64)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, os.path.join(opt.dataroot, 'High-Res', 'test.nii'))
    nib.save(img, os.path.join(opt.dataroot, 'Low-Res', 'lowres_test.nii'))
    
    # Run training loop
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    
    model = create_model(opt)
    visualizer = Visualizer(opt)
    
    total_steps = 0
    
    for epoch in range(1, 2):
        epoch_start_time = time.time()
        epoch_iter = 0
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += total_steps
            
            model.set_input(data)
            model.optimize_parameters()
            
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.display_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch)
                
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
                    
            if total_steps % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save('latest')
                model.save(epoch)
                
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

if __name__ == '__main__':
    run_dummy_training()
