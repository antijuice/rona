"""Advancing ``dp/dt = A p`` when ``A`` spans seven decades of rate.

Explicit methods cannot do this.  Krylov with a modest subspace is limited by
``tau * ||A||``, and ``||A||`` here is the base-pair zipping rate, ~10^7 s^-1,
while the step we want is a millisecond: ``tau ||A|| ~ 10^4``, and
``expm(tau * H)`` overflows outright.  The stiffness that made trajectory
sampling hopeless does not go away by switching to the master equation - it
moves into the linear algebra, and has to be met there.

Backward Euler meets it.  For a generator, ``(I - dt A)`` is an M-matrix -
positive diagonal, non-positive off-diagonal, weakly diagonally dominant - so

* the scheme is **L-stable**: fast modes are damped towards their equilibrium
  rather than oscillating, which is the physically right behaviour and is what
  the trapezoid rule fails to do;
* the solve is unconditionally stable, so the step is limited by accuracy on the
  *slow* modes, not by the fastest rate in the matrix;
* probabilities stay non-negative, because the inverse of an M-matrix is
  non-negative - a property no explicit scheme gives for free.

It is first order, so accuracy comes from step-doubling with Richardson
extrapolation, which also supplies the error estimate.

That estimate is an *estimate*.  It is not the FSP certificate and must not be
confused with it: the certificate bounds the error from truncating the state
space, and says nothing about the error from integrating in time.  Conflating
the two is a mistake this module's author has already made once, and it showed
up as a reported bound of 3e-11 on an answer that was 7e-2 wrong.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import splu


def _factor(matrix: csr_matrix, dt: float):
    """LU of ``(I - dt A)``, which every substep at this step size reuses.

    Factorising inside the substep loop - which an earlier version did - repeats
    the expensive half of the work for every one of up to 64 identical solves.
    """
    size = matrix.shape[0]
    return splu((identity(size, format="csc") - dt * matrix).tocsc())


def _march(factorisation, vector: np.ndarray, steps: int) -> np.ndarray:
    current = vector
    for _ in range(steps):
        current = np.maximum(factorisation.solve(current), 0.0)
    return current


def integrate(
    matrix: csr_matrix,
    vector: np.ndarray,
    t: float,
    *,
    tolerance: float = 1e-6,
    max_halvings: int = 6,
) -> tuple[np.ndarray, float]:
    """``exp(t A) v`` by backward Euler, refined a bounded number of times.

    Returns the Richardson-extrapolated solution and an estimate of its L1
    error: the step is halved until one step and two half-steps agree to
    ``tolerance`` *or* ``max_halvings`` is reached, and their difference
    estimates the error of the coarser one.

    The refinement is deliberately bounded.  Insisting on convergence would mean
    resolving the fastest mode in the matrix - base-pair zipping at 10^-7 s - to
    answer a question posed at millisecond resolution.  L-stability already puts
    those modes at their equilibrium; refining until they are *resolved* spends
    unbounded work to change nothing.  So the routine reports the accuracy it
    reached rather than guaranteeing the one it was asked for, and the caller is
    told which.
    """
    if t <= 0.0:
        return vector.copy(), 0.0
    coarse = _march(_factor(matrix, t), vector, 1)
    steps = 1
    error = float("inf")
    for _ in range(max_halvings):
        steps *= 2
        fine = _march(_factor(matrix, t / steps), vector, steps)
        error = float(np.abs(fine - coarse).sum())
        if error <= tolerance:
            break
        coarse = fine
    # Richardson: backward Euler is first order, so 2*fine - coarse is second
    extrapolated = np.maximum(2.0 * fine - coarse, 0.0)
    return extrapolated, error
