# -- coding: utf-8 --

"""Utils/MortonEncoding: CUDA-accelerated morton code computation for 3d points."""

import torch
from . import _C

def morton_encode(positions: torch.Tensor) -> torch.Tensor:
    minimum_coordinates = positions.min(dim=0).values
    cube_size = (positions.max(dim=0).values - minimum_coordinates).max()
    return _C.morton_encode_cuda(positions, minimum_coordinates, cube_size)