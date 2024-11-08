"""
Potential base class

Code adapted from https://github.com/vishwakftw/metropolis-adjusted-MLA
"""

import torch

torch.set_default_dtype(torch.float64)


class Potential:
    """
    Base class for Potentials
    """

    def __init__(self, *args, **kwargs):
        pass

    def value(self, x: torch.Tensor):
        raise NotImplementedError

    def gradient(self, x: torch.Tensor):
        raise NotImplementedError
