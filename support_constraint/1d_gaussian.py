"""
Sampling from a 1D truncated Gaussian
"""

import os
import sys
import time

import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jax.nn import relu
import numpy as np

CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import pdlmc_run_chain, projlmc_run_chain


ITERATIONS = int(5e6)

# Constraints
a, b = 1, 3
f = jit(lambda x: jnp.inner(x, x) / 2.0)
g = jit(lambda x: jnp.array(4 * relu(jnp.inner((x - a), (x - b))) - 0.005))
proj = jit(lambda x: jnp.clip(x, a, b))

# Proj. LMC
start = time.process_time()
projlmc_traj = projlmc_run_chain(jr.PRNGKey(1234), f, proj, ITERATIONS, 0.001, jnp.zeros(2))
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
    step_size_lmbda=1e-3,
    step_size_nu=0,
    initial_x=jnp.array(0.0),
    initial_lmbda=jnp.array(0.0),
    initial_nu=jnp.array(0.0),
)
print(f"PD-LMC finished processing in {time.process_time()-start} seconds.")

# Save results
np.savez(
    os.path.join(CWD, "1d_gaussian"),
    projlmc_x=projlmc_traj.x,
    pdlmc_x=pdlmc_traj.x,
    pdlmc_lambda=pdlmc_traj.lmbda,
    pdlmc_nu=pdlmc_traj.nu,
)
