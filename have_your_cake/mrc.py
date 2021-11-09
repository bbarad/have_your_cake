from __future__ import annotations
import mrcfile
import torch
from torch import Tensor
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def read_mrc(filename: Path):
    with mrcfile.open(filename) as mrc:
        volume = np.copy(mrc.data)
    return Tensor(volume)


