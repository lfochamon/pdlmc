"""
Sampling from a 2D truncated Gaussian (mirror LMC)

Code adapted from https://github.com/vishwakftw/metropolis-adjusted-MLA
"""

import os

import torch
import numpy as np

from baselines import MirrorLangevinSampler
from potentials import Potential
from barriers import EllipsoidBarrier

CWD = os.path.dirname(__file__)


class GaussianPotential(Potential):

    def __init__(self, mean: torch.Tensor):
        self.mean = mean
        self.dimension = mean.shape[0]

    def value(self, x: torch.Tensor):
        return torch.inner(x - self.mean, x - self.mean) / 2

    def gradient(self, x: torch.Tensor):
        return 2 * (x - self.mean) / 2


ITERATIONS = int(5e6)

barrier = EllipsoidBarrier(ellipsoid={"rot": torch.eye(2), "eigvals": 1 * torch.ones(2)})
potential = GaussianPotential(2 * torch.ones(2))
sampler = MirrorLangevinSampler(barrier=barrier, potential=potential, num_samples=1)

initial_particles = torch.randn(1, 2)
initial_particles /= torch.linalg.norm(initial_particles, dim=-1, keepdim=True)
initial_particles *= torch.rand(1, 1) * 0.1
sampler.set_initial_particles(initial_particles)

particles = sampler.mix(
    num_iters=ITERATIONS, stepsize=1e-3, return_particles=True, no_progress=False
)
particles_np = particles.numpy().squeeze()

np.savez(os.path.join(CWD, "2d_mirror"), mirror_x=particles_np)
