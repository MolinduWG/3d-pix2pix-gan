import numpy as np
import os

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed. Please install it using 'pip install nibabel'")
    nib = None

def load_nifti(path):
    """
    Load a NIfTI file and return the data as a numpy array and the affine matrix.
    """
    if nib is None:
        raise ImportError("nibabel is not installed.")
    
    img = nib.load(path)
    data = img.get_fdata()
    affine = img.affine
    return data, affine

def save_nifti(data, path, affine):
    """
    Save a numpy array as a NIfTI file.
    """
    if nib is None:
        raise ImportError("nibabel is not installed.")
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

def get_patch_coords(shape, patch_size, stride):
    """
    Generate coordinates for 3D patches.
    shape: tuple of (d, h, w)
    patch_size: tuple of (pd, ph, pw)
    stride: tuple of (sd, sh, sw)
    Returns a list of (z, y, x) coordinates.
    """
    d, h, w = shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    coords = []
    
    # Calculate number of patches in each dimension
    nz = (d - pd) // sd + 1
    ny = (h - ph) // sh + 1
    nx = (w - pw) // sw + 1
    
    # Handle edge cases where the volume is not perfectly divisible by stride
    # We will ensure we cover the whole volume, potentially overlapping more at the end
    
    z_coords = list(range(0, d - pd + 1, sd))
    if z_coords[-1] + pd < d:
        z_coords.append(d - pd)
        
    y_coords = list(range(0, h - ph + 1, sh))
    if y_coords[-1] + ph < h:
        y_coords.append(h - ph)
        
    x_coords = list(range(0, w - pw + 1, sw))
    if x_coords[-1] + pw < w:
        x_coords.append(w - pw)
        
    for z in z_coords:
        for y in y_coords:
            for x in x_coords:
                coords.append((z, y, x))
                
    return coords

def extract_patch(volume, coord, patch_size):
    """
    Extract a patch from the volume at the given coordinate.
    """
    z, y, x = coord
    pd, ph, pw = patch_size
    return volume[z:z+pd, y:y+ph, x:x+pw]

def stitch_volume(patches, coords, output_shape, patch_size):
    """
    Stitch patches back into a volume.
    Averages overlapping regions.
    """
    output_volume = np.zeros(output_shape)
    count_volume = np.zeros(output_shape)
    
    pd, ph, pw = patch_size
    
    for patch, (z, y, x) in zip(patches, coords):
        output_volume[z:z+pd, y:y+ph, x:x+pw] += patch
        count_volume[z:z+pd, y:y+ph, x:x+pw] += 1
        
    # Avoid division by zero
    count_volume[count_volume == 0] = 1
    
    return output_volume / count_volume
