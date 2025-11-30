import torch
from models import networks3d

def test_unet_generator():
    input_nc = 1
    output_nc = 1
    ngf = 64
    norm = 'batch'
    use_dropout = False
    gpu_ids = []
    
    netG = networks3d.define_G(input_nc, output_nc, ngf, 'unet_128', norm, use_dropout, gpu_ids)
    print("Generator created successfully")
    
    # 64x64x64 patch
    input_tensor = torch.randn(1, input_nc, 64, 64, 64)
    output = netG(input_tensor)
    print("Forward pass successful")
    print("Output shape:", output.shape)

if __name__ == '__main__':
    test_unet_generator()
