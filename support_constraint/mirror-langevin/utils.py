"""
Utils (Cholesky decomposition)

Code adapted from https://github.com/vishwakftw/metropolis-adjusted-MLA
"""

import torch

torch.set_default_dtype(torch.float64)

from typing import Callable


def get_chol(is_diagonal: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    if is_diagonal:

        def CHOL(batch_of_matrices: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(batch_of_matrices)  # return diagonal

    else:

        def CHOL(batch_of_matrices: torch.Tensor) -> torch.Tensor:
            L, info = torch.linalg.cholesky_ex(batch_of_matrices, upper=False)
            if info.sum(0) == 0:
                return L

            max_iter = 3  # 3 maximum perturbations
            curr_pert = 1e-06  # perturbation values, increased multiplicatively
            mask = info.to(dtype=torch.bool)
            # we only need to deal with the failed ones below
            copy_failed = batch_of_matrices[mask].clone()
            while info.sum() != 0 and max_iter > 0:
                # do inplace to save memory
                copy_failed.diagonal(dim1=-2, dim2=-1).add_(curr_pert)
                # perform cholesky on the perturbed matrices
                Lfail, info = torch.linalg.cholesky_ex(
                    copy_failed, check_errors=max_iter == 1, upper=False
                )
                curr_pert *= 10  # increase curr_pert by 10 factor
                max_iter -= 1  # reduce max_iter
            # at the end, if there's no erroring out, replace the failed ones
            L[mask] = Lfail
            return L

    return CHOL
