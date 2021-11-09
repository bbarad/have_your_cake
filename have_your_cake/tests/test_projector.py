from torch import Tensor

from have_your_cake.projector import forward_fft, central_slice
import torch


def test_forward_fft(volume):
    volume_ft = forward_fft(volume)
    assert isinstance(volume_ft, Tensor)
    assert volume_ft.shape == (128, 128, 128)
    assert volume_ft.dtype == torch.complex64


def test_central_slice(volume):
    volume_ft = forward_fft(volume)
    slice_coords = central_slice(volume_ft)
    print(slice_coords)