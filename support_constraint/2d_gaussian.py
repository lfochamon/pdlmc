"""
Sampling from a 2D truncated Gaussian
"""

import os
import sys
import time

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit
from jax.nn import relu

CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import pdlmc_run_chain, projlmc_run_chain


ITERATIONS = int(5e6)
BURN_IN = int(4e6)

# Constraints
f = jit(lambda x: jnp.inner(x - 2 * jnp.ones(2), x - 2 * jnp.ones(2)) / 0.5)
g = jit(lambda x: jnp.array(relu(jnp.inner(x, x) - 1.0) - 0.001))
proj = jit(lambda x: x / jnp.maximum(jnp.sqrt(jnp.inner(x, x)), 1.0))


# Rejection sampling
def rejection(N):
    proposal = np.random.normal(loc=2.0, size=(100000, 2))
    dists = (proposal * proposal).sum(axis=1)
    samples = proposal[dists <= 1.0]

    while samples.shape[0] <= N:
        proposal = np.random.normal(loc=2.0, size=(100000, 2))
        dists = (proposal * proposal).sum(axis=1)
        samples = np.append(samples, proposal[dists <= 1.0], axis=0)

    return samples[:N]


start = time.process_time()
rejection_samples = rejection(BURN_IN)
print(f"Rejection sampling finished processing in {time.process_time()-start} seconds.")

# Proj. LMC
start = time.process_time()
projlmc_traj = projlmc_run_chain(jr.PRNGKey(1234), f, proj, ITERATIONS, 1e-3, jnp.zeros(2))
print(f"Proj-LMC finished processing in {time.process_time()-start} seconds.")

# PD-LMC
start = time.process_time()
_, pdlmc_traj = pdlmc_run_chain(
    initial_key=jr.PRNGKey(1234),
    f=f,
    g=g,
    h=lambda _: 0,
    iterations=ITERATIONS,
    lmc_steps=1,
    burnin=0,
    step_size_x=1e-3,
    step_size_lmbda=2e-1,
    step_size_nu=0,
    initial_x=jnp.zeros(2),
    initial_lmbda=jnp.array(0.0),
    initial_nu=jnp.array(0.0),
)
print(f"PD-LMC finished processing in {time.process_time()-start} seconds.")

np.savez(
    os.path.join(CWD, "2d_gaussian"),
    projlmc_x=projlmc_traj.x,
    pdlmc_x=pdlmc_traj.x,
    pdlmc_lambda=pdlmc_traj.lmbda,
    pdlmc_nu=pdlmc_traj.nu,
    rejection_x=rejection_samples,
)
