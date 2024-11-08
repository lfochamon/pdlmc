"""
Counterfactual sampling: (exactly) 20% higher (log-)returns on NVDA and LLY stocks
"""

import os
import sys
from collections import namedtuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import pandas as pd
from jax import jit
from jax.tree_util import Partial

CWD = os.path.dirname(__file__)
sys.path.append(os.path.join(CWD, ".."))
from pdlmc import projlmc_run_chain, pdlmc_run_chain


np.set_printoptions(precision=4)
MuSigma = namedtuple("MuSigma", ["mu", "sigma"])
ITERATIONS = int(4e5)

# Read stock time-series and create the (log-)returns matrix
stock_files = {}
stocks_label = ["TSLA", "NVDA", "JNJ", "AAPL", "GOOG", "BRK-B", "LLY"]
for f in [os.path.join(CWD, f"data/{stock}.csv") for stock in stocks_label]:
    stock_files[os.path.basename(f)[: -len(".csv")]] = pd.read_csv(f, index_col="Date")["Adj Close"]

stocks = pd.DataFrame(stock_files)
stocks = 100 * np.log(stocks).diff()
stocks.dropna(inplace=True)
n_stocks = stocks.shape[1]
print(stocks.info())


@jit
def inv_wishart_logpdf(X, Psi, nu):
    return -0.5 * (nu + n_stocks + 1) * jnp.linalg.slogdet(X)[1] - 0.5 * jnp.trace(
        jnp.dot(Psi, jnp.linalg.inv(X))
    )


@jit
def logpdf_prior(musig: MuSigma, tau=3.0, nu=12.0):
    Psi = np.eye(n_stocks)

    return jsp.stats.multivariate_normal.logpdf(
        x=musig.mu, mean=np.zeros(n_stocks), cov=tau * np.eye(n_stocks)
    ) + inv_wishart_logpdf(musig.sigma, Psi, nu)


@jit
def log_likelihood(musig: MuSigma):
    return jnp.sum(
        jsp.stats.multivariate_normal.logpdf(x=stocks.to_numpy(), mean=musig.mu, cov=musig.sigma)
    )


@jit
def neglogpdf_posterior(musig: MuSigma):
    return -logpdf_prior(musig) - log_likelihood(musig)


@jit
def _eqconst(musig: MuSigma, w, rhostar):
    return jnp.array([jnp.inner(musig.mu, w) - rhostar]).squeeze()


# LMC
init_lmc = MuSigma(jnp.zeros(n_stocks), 10 * jnp.eye(n_stocks))

lmc_traj = projlmc_run_chain(
    jr.PRNGKey(123),
    neglogpdf_posterior,
    lambda x: MuSigma(x.mu, 0.5 * (x.sigma + x.sigma.T)),
    ITERATIONS,
    1e-3,
    init_lmc,
)


# PD-LMC
init_pdlmc = MuSigma(jnp.zeros(n_stocks), 10 * jnp.eye(n_stocks))

mu_hat = stocks.mean()
idx_stocks = ["NVDA", "LLY"]
mask = jnp.array([jnp.eye(n_stocks)[stocks.columns == stock, :].squeeze() for stock in idx_stocks])
desired_vals = jnp.array(
    [mu_hat[stock] * (1.2 if mu_hat[stock] >= 0 else 0.8) for stock in idx_stocks]
)
eqconst = Partial(_eqconst, w=mask, rhostar=desired_vals)

_, pdlmc_traj = pdlmc_run_chain(
    jr.PRNGKey(123),
    neglogpdf_posterior,
    lambda _: 0,
    eqconst,
    ITERATIONS,
    1,
    0,
    1e-3,
    0,
    6e-3,
    init_pdlmc,
    jnp.array(0),
    jnp.zeros(len(idx_stocks)).squeeze(),
    lambda x: MuSigma(x.mu, 0.5 * (x.sigma + x.sigma.T)),
)


np.savez(
    os.path.join(CWD, "nvda_lly"),
    pdlmc_mu=pdlmc_traj.x.mu,
    pdlmc_sigma=pdlmc_traj.x.sigma,
    pdlmc_lambda=pdlmc_traj.lmbda,
    pdlmc_nu=pdlmc_traj.nu,
    lmc_mu=lmc_traj.x.mu,
    lmc_sigma=lmc_traj.x.sigma,
)
