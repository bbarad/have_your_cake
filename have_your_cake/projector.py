import numpy as np
import torch
from torch import Tensor
import einops
from scipy.interpolate import interpn
from scipy.spatial.transform import Rotation


side = 128
volume = np.random.rand(128,128,128)
volume = Tensor(volume)
print(volume.shape)


def forward_fft(real_volume):
        """Take a real volume as a 3D tensor and return an RFFT of that tensor"""
        forward = torch.fft.fftn(real_volume)
        forward_shifted = torch.fft.fftshift(forward)
        return forward_shifted


def central_slice(fft_volume, orientation=Tensor((0,0,0))):
        """Take an fft volume and an orientation and return a central slice
        fft_volume: 3D FFT'd volume as a torch tensor
        orientation: Euler angles in a torch tensor
        """
        length = fft_volume.shape[0]
        x = torch.linspace(-length / 2, (length / 2) - 1)
        y = x
        x_coords, y_coords = torch.meshgrid(x, y)
        z_coords = torch.zeros_like(x_coords)
        slice_coords = torch.stack((x_coords, y_coords, z_coords), axis=0)  # (3, 128, 128)
        slice_coords_flattened = einops.rearrange(slice_coords, 'pos w h -> (w h) pos')

        rotation = Rotation.from_euler(seq='ZYZ', angles=orientation)
        rotated_slice_coords = rotation.apply(slice_coords_flattened)
        return rotated_slice_coords












