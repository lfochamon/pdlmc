"""
Barriers and mirror maps

Code adapted from https://github.com/vishwakftw/metropolis-adjusted-MLA
"""

import torch

torch.set_default_dtype(torch.float64)

from typing import Dict


class Barrier:
    """
    Base class for Barriers
    """

    diag_hess = False
    ZERO = torch.tensor(0.0)

    def __init__(self, *args, **kwargs):
        pass

    def feasibility(self, x: torch.Tensor):
        """
        Returns if x is feasible.
        """
        raise NotImplementedError

    def value(self, x: torch.Tensor):
        raise NotImplementedError

    def gradient(self, x: torch.Tensor):
        raise NotImplementedError

    def inverse_gradient(self, y: torch.Tensor):
        raise NotImplementedError

    def hessian(self, x: torch.Tensor):
        raise NotImplementedError

    def sample_uniform(self, n_points: int):
        raise NotImplementedError

    def boundary_to_interior_half(self, x: torch.Tensor):
        raise NotImplementedError


class BoxBarrier(Barrier):
    """
    Log barrier of box
    """

    def __init__(self, bounds: torch.Tensor):
        self.bounds = bounds  # [a1, a2, ...] form the box [-a1, a1] x [-a2, a2] x ...
        self.dimension = bounds.shape[-1]
        self.diag_hess = True

    def _safe_diff(self, x: torch.Tensor):
        """
        Returns bounds ** 2 - x ** 2, but with care around the boundary
        """
        return torch.clamp_min(torch.square(self.bounds) - torch.square(x), min=1e-08)

    def feasibility(self, x: torch.Tensor):
        """
        Returns if x is feasible.
        """
        return torch.all(torch.abs(x) <= self.bounds, dim=-1)

    def value(self, x: torch.Tensor):
        """
        Computes the value of the potential at x
        defined as -log(1 - <x, Ax>) * c
        where A is the ellipsoid and c is the inverse temperature
        """
        return -torch.sum(torch.log(self._safe_diff(x)), dim=-1)

    def gradient(self, x: torch.Tensor):
        """
        Computes the gradient of the potential at x
        defined as 2 * x_{i} / (a_{i} ** 2 - x_{i} ** 2)
        """
        return 2 * x / self._safe_diff(x)

    def inverse_gradient(self, y: torch.Tensor):
        """
        Computes the inverse of the gradient of the potential at y.
        Element-wise defined as (-1 + sqrt(1 + a[i]^2 y[i]^2)) / y[i].
        """
        asqr_ysqr = torch.square(self.bounds * y)
        raw = (-1 + torch.sqrt(1 + asqr_ysqr)) / y
        raw[torch.isclose(y, self.ZERO, atol=1e-07, rtol=1e-05)] = 0  # if very close to 0, then 0
        return raw

    def hessian(self, x: torch.Tensor):
        """
        Computes the diagonal of the hessian (in this case is diagonal)
        """
        reci_diff = self._safe_diff(x).reciprocal_()
        output = 2 * reci_diff + 4 * torch.square(x * reci_diff)
        return output.clamp_max_(1e07)

    def sample_uniform(self, n_points: int):
        """
        Sampling points uniformly from a box
        """
        return torch.rand(n_points, self.dimension) * (2 * self.bounds) - self.bounds

    def boundary_to_interior_half(self, x: torch.Tensor):
        """
        Returns if x belongs in the region between the
        boundary and the interior where the volume of this
        region is 0.5 under the uniform distribution
        """
        return torch.any(torch.abs(x) > self.bounds * 0.5 ** (1 / self.dimension), dim=-1)


class EllipsoidBarrier(Barrier):
    """
    Log barrier for Ellipsoid
    """

    def __init__(self, ellipsoid: Dict[str, torch.Tensor]):
        self.ellipsoid = ellipsoid
        self.dimension = ellipsoid["eigvals"].shape[0]

    def _ellipsoid_inner_product(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes <x, Ax> efficiently.
        """
        UTx = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, x)
        return torch.sum(torch.square(UTx) * self.ellipsoid["eigvals"], dim=-1)

    def _inverse_ellipsoid_inner_product(self, y: torch.Tensor):
        """
        Computes <y, inv(A)y> efficiently.
        """
        UTy = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, y)
        return torch.sum(torch.square(UTy) / self.ellipsoid["eigvals"], dim=-1)

    def _ellipsoid_map(self, x: torch.Tensor):
        """
        Computes Ax efficiently.
        """
        UTx = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, x)
        return torch.einsum("ij,...j->...i", self.ellipsoid["rot"] * self.ellipsoid["eigvals"], UTx)

    def _inverse_ellipsoid_map(self, y: torch.Tensor):
        """
        Computes inv(A)y efficiently.
        """
        UTx = torch.einsum("ij,...j->...i", self.ellipsoid["rot"].T, y)
        return torch.einsum("ij,...j->...i", self.ellipsoid["rot"] / self.ellipsoid["eigvals"], UTx)

    def feasibility(self, x: torch.Tensor):
        """
        Returns if x is feasible.
        """
        return self._ellipsoid_inner_product(x) <= 1

    def value(self, x: torch.Tensor):
        """
        Computes the value of the potential at x
        defined as -log(1 - <x, Ax>) * c
        where A is the ellipsoid and c is the inverse temperature
        """
        inner_product = self._ellipsoid_inner_product(x)
        inner_product[inner_product >= 1] = 0  # points outside or on the boundary
        return -torch.log1p(-inner_product)

    def gradient(self, x: torch.Tensor):
        """
        Computes the gradient of the potential at x
        defined as 2 * c * (Ax) / (1 - <x, Ax>)
        where A is the ellipsoid and c is the inverse temperature
        """
        one_minus_xAx = 1 - self._ellipsoid_inner_product(x)
        # NOTE: when x is too close to the boundary
        # this might be really small and cause problems for us when
        # dividing by 1 - <x, Ax>.
        one_minus_xAx.clamp_min_(min=1e-08)
        Ax = self._ellipsoid_map(x)
        return 2 * Ax / one_minus_xAx.unsqueeze(dim=-1)

    def inverse_gradient(self, y: torch.Tensor):
        """
        Computes the inverse of the potential at y
        defined as x which satisfies gradient(x) = y.
        This is also equal to the gradient of the Fenchel-Legendre dual.
        """
        # We did some complicated calculations for the inverse of the function
        # g(x) = x / (1 - <x, Ax>).
        # inv(g)(y) = lambda(y) * y, where lambda(y) = (-1 + sqrt(4<y, Ay> + 1)) / (2<y, Ay>)
        # we are interested in the inverse of the function h(x) = 2cAx / (1 - <x, Ax>)
        # note that h(x) = j(g(x)), where j(z) = 2cAz.
        # therefore, inv(h)(y) = inv(g)(inv(j)(y))
        # NOTE: if <y, inv(A)y> is too small, or 0, we'll run into numerical issues.
        # so, we're artificially clipping it.
        yinvAy = torch.clamp_min(self._inverse_ellipsoid_inner_product(y), min=1e-08)
        lambda_cons = (-1 + torch.sqrt(1 + yinvAy)) / yinvAy
        return lambda_cons.unsqueeze(dim=-1) * self._inverse_ellipsoid_map(y)

    def hessian(self, x: torch.Tensor):
        """
        Computes the square root of the Hessian of the potential at x
        defined as c * (2A / (1 - <x, Ax>) + 4 (Ax)(Ax).T / (1 - <x, Ax>)^2).
        This has a nice structure which we could exploit for the square root,
        but not doing it here.
        """
        # NOTE: when x is too close to the boundary
        # this might be really small and cause problems for us when
        # dividing by 1 - <x, Ax>.
        one_minus_xAx = 1 - self._ellipsoid_inner_product(x)  # batchsize
        one_minus_xAx.clamp_min_(min=1e-08)
        Ax = self._ellipsoid_map(x)
        U = self.ellipsoid["rot"]
        L = self.ellipsoid["eigvals"]
        scaled_Ax = 2 * Ax / one_minus_xAx.unsqueeze_(dim=-1)
        scaled_L = 2 * L / one_minus_xAx.unsqueeze_(dim=-1)
        ULUT = torch.einsum("...ij,jk->...ik", U * scaled_L, U.T)
        return ULUT + scaled_Ax.unsqueeze(dim=-1) * scaled_Ax.unsqueeze(dim=-2)

    def sample_uniform(self, n_points: int):
        """
        Sampling points uniformly from an ellipsoid.
        This is just rejection sampling, so is very inefficient in higher dimensions
        volume of cuboid with side lengths [2/a_{i}]_{i = 1}^{d} is 2^{d} / prod([a_{i}]_{i = 1}^{d})
        volume of ellipsoid with axis lengths [1/a_{i}] is \pi / prod([a_{i}]^{i = 1}^{d})
        So there's a pi /2^{d} chance that a point sampled within a cuboid lies in the ellipsoid
        """
        dimension = self.ellipsoid["eigvals"].shape[0]
        fraction = 2**dimension / torch.pi
        all_points = torch.empty(0, dimension)
        while all_points.shape[0] < n_points:
            # a little excess just in case
            points = torch.rand(int(1.1 * fraction * n_points), dimension) * 2 - 1
            points = torch.einsum("ij,...i->...j", self.ellipsoid["rot"], points)
            points = points * torch.sqrt(self.ellipsoid["eigvals"])
            feasible = self.feasibility(points)
            all_points = torch.vstack([all_points, points[feasible]])
        return all_points[:n_points]

    def boundary_to_interior_half(self, x: torch.Tensor):
        """
        Returns if x belongs in the region between the
        boundary and the interior where the volume of this
        region is 0.5 under the uniform distribution
        """
        return self._ellipsoid_inner_product(x) > (0.5 ** (2 / self.dimension))
