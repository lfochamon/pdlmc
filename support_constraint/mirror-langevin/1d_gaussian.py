"""
Sampling from a 1D truncated Gaussian (mirror LMC)

Code adapted from https://github.com/vishwakftw/metropolis-adjusted-MLA
"""

import os

import torch
import numpy as np

from baselines import MirrorLangevinSampler
from potentials import Potential
from barriers import BoxBarrier

CWD = os.path.dirname(__file__)


class GaussianPotential(Potential):

    def __init__(self, mean: torch.Tensor):
        self.mean = mean
        self.dimension = mean.shape[0]

    def value(self, x: torch.Tensor):
        return torch.inner(x - self.mean, x - self.mean) / 2

    def gradient(self, x: torch.Tensor):
        return 2 * (x - self.mean) / 2


a, b = 1, 3
ITERATIONS = int(5e6)

# Barrier is symmetric [-c,c]. Using the translation equivariance of the Gaussian,
# x distributed according to N(0,1) truncated to [a,b] has the same distribution as
# y+(a+b)/2 for y distributed according N(-(a+b)/2,1) truncated to [-(b-a)/2,(b-a)/2]
barrier = BoxBarrier(bounds=(b - a) / 2 * torch.ones(1))
potential = GaussianPotential(-(a + b) / 2 * torch.ones(1))
sampler = MirrorLangevinSampler(barrier=barrier, potential=potential, num_samples=1)

initial_particles = torch.rand(1)
initial_particles = (b - a) / 4 * (initial_particles - 0.5)
sampler.set_initial_particles(initial_particles)

particles = sampler.mix(
    num_iters=ITERATIONS, stepsize=1e-3, return_particles=True, no_progress=False
)
particles_np = particles.numpy().squeeze() + (a + b) / 2

np.savez(os.path.join(CWD, "1d_mirror"), mirror_x=particles_np)
