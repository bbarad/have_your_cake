from have_your_cake.mrc import read_mrc
import torch
from torch import Tensor


def test_read_mrc(mrc_file):
    volume = read_mrc(mrc_file)
    assert isinstance(volume, Tensor)
    assert volume.shape == (128, 128, 128)