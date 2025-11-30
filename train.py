import time
import os
import util.util as util
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()

    # Save images to results directory
    if epoch % opt.display_freq == 0:
        save_result_dir = os.path.join(opt.results_dir, opt.name, 'train_%d' % epoch)
        util.mkdirs(save_result_dir)
        visuals = model.get_current_visuals()
        for label, image_numpy in visuals.items():
            img_path = os.path.join(save_result_dir, '%s.png' % label)
            util.save_image(image_numpy, img_path)

print('saving the final model')
model.save('final')

# Generate and save loss plots
print('generating loss plots...')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

log_file = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
if os.path.exists(log_file):
    epochs = []
    g_gan_losses = []
    g_l1_losses = []
    d_real_losses = []
    d_fake_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('(epoch:'):
                match = re.search(r'epoch: (\d+).*G_GAN: ([\d.]+).*G_L1: ([\d.]+).*D_Real: ([\d.]+).*D_Fake: ([\d.]+)', line)
                if match:
                    epochs.append(int(match.group(1)))
                    g_gan_losses.append(float(match.group(2)))
                    g_l1_losses.append(float(match.group(3)))
                    d_real_losses.append(float(match.group(4)))
                    d_fake_losses.append(float(match.group(5)))
    
    if epochs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(epochs, g_gan_losses, 'b-', alpha=0.7)
        axes[0, 0].set_title('Generator GAN Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, g_l1_losses, 'g-', alpha=0.7)
        axes[0, 1].set_title('Generator L1 Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, d_real_losses, 'r-', alpha=0.7)
        axes[1, 0].set_title('Discriminator Real Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, d_fake_losses, 'orange', alpha=0.7)
        axes[1, 1].set_title('Discriminator Fake Loss')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(opt.checkpoints_dir, opt.name, 'loss_plots.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Loss plots saved to {plot_path}')

