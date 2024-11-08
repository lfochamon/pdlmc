"""
Sampling from a 1D truncated Gaussian (different mini-batch sizes)
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
from pdlmc import pdlmc_run_chain


INTERATIONS = int(5e5)

# Constraints
a, b = 1, 3
f = jit(lambda x: jnp.inner(x, x) / 2.0)
g = jit(lambda x: jnp.array(4 * relu(jnp.inner((x - a), (x - b))) - 0.005))

pdlmc_traj = {}
for n in [1, 10, 100, 1000]:
    start = time.process_time()
    _, pdlmc_traj[n] = pdlmc_run_chain(
        initial_key=jr.PRNGKey(1234),
        f=f,
        g=g,
        h=lambda _: 0,
        iterations=INTERATIONS,
        lmc_steps=1,
        burnin=0,
        step_size_x=1e-3,
        step_size_lmbda=1e-3,
        step_size_nu=0,
        initial_x=jnp.array(0.0),
        initial_lmbda=jnp.array(0.0),
        initial_nu=jnp.array(0.0),
    )
    print(f"PD-LMC (N = {n}) finished processing in {time.process_time()-start} seconds.")

np.savez(
    os.path.join(CWD, "1d_gaussian_vs_N"),
    pdlmc_x_1=pdlmc_traj[1].x,
    pdlmc_lambda_1=pdlmc_traj[1].lmbda,
    pdlmc_nu_1=pdlmc_traj[1].nu,
    pdlmc_x_10=pdlmc_traj[10].x,
    pdlmc_lambda_10=pdlmc_traj[10].lmbda,
    pdlmc_nu_10=pdlmc_traj[10].nu,
    pdlmc_x_100=pdlmc_traj[100].x,
    pdlmc_lambda_100=pdlmc_traj[100].lmbda,
    pdlmc_nu_100=pdlmc_traj[100].nu,
    pdlmc_x_1000=pdlmc_traj[1000].x,
    pdlmc_lambda_1000=pdlmc_traj[1000].lmbda,
    pdlmc_nu_1000=pdlmc_traj[1000].nu,
)
