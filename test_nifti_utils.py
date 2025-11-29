import unittest
import numpy as np
import sys
import os

# Mock nibabel if not present
try:
    import nibabel
except ImportError:
    from unittest.mock import MagicMock
    sys.modules['nibabel'] = MagicMock()

from util import nifti_utils

class TestNiftiUtils(unittest.TestCase):
    def test_get_patch_coords(self):
        shape = (100, 100, 100)
        patch_size = (32, 32, 32)
        stride = (32, 32, 32)
        
        coords = nifti_utils.get_patch_coords(shape, patch_size, stride)
        
        # Check if we cover the whole volume
        # Simple check: last coord + patch size >= shape
        last_coord = coords[-1]
        self.assertTrue(last_coord[0] + patch_size[0] >= shape[0])
        self.assertTrue(last_coord[1] + patch_size[1] >= shape[1])
        self.assertTrue(last_coord[2] + patch_size[2] >= shape[2])
        
    def test_patch_stitch_consistency(self):
        shape = (64, 64, 64)
        patch_size = (32, 32, 32)
        stride = (16, 16, 16) # Overlapping
        
        # Create random volume
        volume = np.random.rand(*shape)
        
        coords = nifti_utils.get_patch_coords(shape, patch_size, stride)
        
        patches = []
        for coord in coords:
            patches.append(nifti_utils.extract_patch(volume, coord, patch_size))
            
        reconstructed = nifti_utils.stitch_volume(patches, coords, shape, patch_size)
        
        # Check if reconstruction matches original
        # Note: overlapping regions are averaged, so it should match exactly if we just slice and put back.
        # But wait, if we average, and the original values are the same, average is same.
        np.testing.assert_allclose(volume, reconstructed, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
