"""``exp(tA)v`` for a stiff sparse generator, by Krylov with adaptive substeps.

scipy's ``expm_multiply`` is scaling-and-squaring, whose cost scales with
``||A|| t``.  Here ``||A||`` is set by the fastest move in the system - zipping a
base pair, ~10^7 s^-1 - and ``t`` is a second, so ``||A|| t ~ 10^7`` and it does
not finish.  The stiffness that made trajectory sampling hopeless does not
vanish by switching to the master equation; it moves into the integrator, and
has to be met there.

Krylov meets it.  The Arnoldi approximation
``exp(tA)v ~ ||v|| V_m exp(tH_m) e_1`` converges according to the part of the
spectrum the starting vector actually explores, not the norm of the whole
matrix, and fast modes that are already equilibrated in ``v`` contribute almost
nothing.  This is the Expokit scheme (Sidje 1998): build a small Krylov basis,
estimate the local error from the first neglected term, accept or shrink the
substep, repeat.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def expmv(
    matrix,
    vector: np.ndarray,
    t: float,
    *,
    krylov: int = 30,
    tolerance: float = 1e-10,
    max_substeps: int = 10_000,
) -> np.ndarray:
    """``exp(t * matrix) @ vector``, for a sparse generator.

    ``tolerance`` is the per-unit-time local error budget; the routine takes as
    many substeps as it needs to stay inside it.  Non-negativity is enforced at
    each substep, which the exact solution of a generator satisfies anyway and
    which keeps rounding from producing negative probabilities.
    """
    if t == 0.0:
        return vector.copy()
    size = vector.shape[0]
    dimension = min(krylov, max(1, size - 1)) if size > 1 else 1
    current = vector.astype(float, copy=True)
    remaining = float(t)
    # a first guess at the substep from the matrix's scale
    scale = abs(matrix).max() if matrix.nnz else 1.0
    tau = min(remaining, 1.0 / scale if scale > 0 else remaining) * dimension
    if tau <= 0.0 or not np.isfinite(tau):
        tau = remaining
    substeps = 0

    while remaining > 0.0 and substeps < max_substeps:
        substeps += 1
        tau = min(tau, remaining)
        beta = float(np.linalg.norm(current))
        if beta == 0.0:
            break

        basis = np.zeros((size, dimension + 1))
        hessenberg = np.zeros((dimension + 1, dimension))
        basis[:, 0] = current / beta
        used = dimension
        breakdown = False
        for j in range(dimension):
            w = matrix @ basis[:, j]
            for i in range(j + 1):
                hessenberg[i, j] = basis[:, i] @ w
                w = w - hessenberg[i, j] * basis[:, i]
            norm = float(np.linalg.norm(w))
            hessenberg[j + 1, j] = norm
            if norm < 1e-14:
                used = j + 1
                breakdown = True
                break
            basis[:, j + 1] = w / norm

        # Expokit's local error estimate: exponentiate the *augmented* matrix,
        # one row bigger than the Krylov basis, and read off the weight that
        # would have landed on the basis vector we did not keep.  Computing it
        # from the un-augmented block and multiplying by tau - which is what an
        # earlier version of this did - understates the error by a factor of
        # tau, so steps are accepted that should not be, and the integrator
        # silently returns an answer a few percent wrong on a stiff generator.
        if breakdown:
            small = expm(tau * hessenberg[:used, :used])
            coefficients = small[:, 0]
            error = 0.0
        else:
            augmented = np.zeros((used + 1, used + 1))
            augmented[:used, :used] = hessenberg[:used, :used]
            augmented[used, used - 1] = hessenberg[used, used - 1]
            small = expm(tau * augmented)
            coefficients = small[:used, 0]
            error = beta * abs(small[used, 0])
        budget = tolerance * (tau / float(t))

        if error <= budget or tau <= 1e-15 * float(t):
            current = np.maximum(beta * (basis[:, :used] @ coefficients), 0.0)
            remaining -= tau
            if error > 0.0:
                # grow the step, conservatively, as Expokit does
                tau *= min(2.0, 0.9 * (budget / error) ** (1.0 / max(used, 1)))
        else:
            tau *= max(0.2, 0.9 * (budget / error) ** (1.0 / max(used, 1)))
    return current
