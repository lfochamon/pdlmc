"""
Sampling from a 2D truncated Gaussian (different mini-batch sizes)
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


ITERATIONS = int(1e5)

# Constraints
f = jit(lambda x: jnp.inner(x - 2 * jnp.ones(2), x - 2 * jnp.ones(2)) / 0.5)
g = jit(lambda x: jnp.array(relu(jnp.inner(x, x) - 1.0) - 0.001))

pdlmc_traj = {}
for n in [1, 10, 100, 1000]:
    start = time.process_time()
    _, pdlmc_traj[n] = pdlmc_run_chain(
        initial_key=jr.PRNGKey(1234),
        f=f,
        g=g,
        h=lambda _: 0,
        iterations=ITERATIONS,
        lmc_steps=n,
        burnin=0,
        step_size_x=1e-3,
        step_size_lmbda=2e-1,
        step_size_nu=0,
        initial_x=jnp.zeros(2),
        initial_lmbda=jnp.array(0),
        initial_nu=jnp.array(0),
    )
    print(f"PD-LMC (N = {n}) finished processing in {time.process_time()-start} seconds.")

np.savez(
    os.path.join(CWD, "2d_gaussian_vs_N"),
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
