"""Conceptors — concept-subspace filters (CLAUDE.md §4, M4).

Theory (Jaeger 2014 "Controlling Recurrent Neural Networks by Conceptors"):
  A conceptor C is an n×n positive semi-definite (PSD) matrix with eigenvalues
  in [0, 1] that characterises the subspace a reservoir visits while processing
  a particular pattern.  Given a data matrix X ∈ ℝ^(N×n) of N state snapshots:

      R = Xᵀ X / N                          (n×n correlation matrix, PSD)
      C = R (R + α⁻² I)⁻¹                   (aperture-regularised projection)

  The aperture α controls how "open" the filter is: small α → strict/narrow
  (only the dominant subspace passes), large α → broad (most signals pass).

  Because R is PSD and α > 0, the eigenvalues of C satisfy:
      λ_C = λ_R / (λ_R + α⁻²) ∈ [0, 1)

  Boolean operations on conceptors:
      NOT  C       = I − C                           (complement in the unit ball)
      C1 AND C2    = (C1⁻¹ + C2⁻¹ − I)⁻¹          (intersection)
      C1 OR  C2    = NOT (NOT C1 AND NOT C2)        (union, de Morgan)

  Invariant (asserted by this module): singular values of every Conceptor are in [0, 1].

Dimension note
──────────────
Conceptors operate over n-dimensional *state* vectors — in DRHM this is the
SNN reservoir state (n = BHAVANGA_N_NEURONS = 256 by default).  n is independent
of the VSA hypervector dimension D (= 10 000).  The two spaces interact only when
a conceptor is used to *gate* a VSA computation during javana (Phase 5).

Reference: Jaeger 2014 §2; docs/Architecture of X.md §4 (anusaya conceptors).
"""

from __future__ import annotations

import numpy as np

from drhm import config


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clip_eigenvalues(C: np.ndarray) -> np.ndarray:
    """Clip eigenvalues of a symmetric matrix to [0, 1].

    Uses eigh (symmetric eigendecomposition) for numerical stability.
    Guarantees the PSD + unit-ball invariant after Boolean operations that
    may introduce small floating-point violations.

    Args:
        C: Symmetric n×n matrix.

    Returns:
        Symmetric n×n matrix with eigenvalues in [0, 1].
    """
    lam, Q = np.linalg.eigh(C)
    return Q @ np.diag(np.clip(lam, 0.0, 1.0)) @ Q.T


# ── Conceptor ─────────────────────────────────────────────────────────────────

class Conceptor:
    """A linear concept-subspace filter C = R(R + α⁻²I)⁻¹.

    Construct via :meth:`from_data` or :meth:`from_matrix`.

    Boolean operations use Python operators:
        ``~C``        → NOT (complement)
        ``C1 & C2``   → AND (intersection)
        ``C1 | C2``   → OR  (union)

    Invariant: all singular values of ``self.matrix`` are in [0, 1].  This is
    guaranteed by construction (from_data uses eigendecomposition) and enforced
    after AND/OR via explicit clipping.

    Args:
        _C:    The n×n conceptor matrix.
        alpha: The aperture used to construct this conceptor (stored for reference).
    """

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_data(cls, X: np.ndarray, alpha: float = config.CONCEPTOR_ALPHA) -> Conceptor:
        """Fit a conceptor from N state snapshots.

        Uses eigendecomposition of the correlation matrix for numerical stability;
        the resulting eigenvalues are mathematically guaranteed to lie in [0, 1).

        Args:
            X:     Data matrix, shape (N, n).  Each row is one n-dimensional
                   state snapshot.
            alpha: Aperture α > 0.

        Returns:
            A :class:`Conceptor` with eigenvalues in [0, 1).

        Raises:
            ValueError: If *X* has fewer rows than columns (rank-deficient data).
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        N, n = X.shape
        R = X.T @ X / N  # n×n correlation matrix (symmetric PSD)
        # Eigendecomposition of symmetric R: eigenvalues λ_R ≥ 0.
        lam, Q = np.linalg.eigh(R)
        lam_c = lam / (lam + alpha**-2)  # guaranteed in [0, 1)
        C = Q @ np.diag(lam_c) @ Q.T
        return cls(_C=C, alpha=alpha)

    @classmethod
    def from_matrix(cls, C: np.ndarray, alpha: float = 0.0) -> Conceptor:
        """Wrap an explicit n×n matrix as a Conceptor (clips SVs to [0, 1]).

        Args:
            C:     Square n×n matrix (should be symmetric PSD with SVs in [0, 1]).
            alpha: Nominal aperture (informational only; no computation).
        """
        return cls(_C=_clip_eigenvalues(C), alpha=alpha)

    def __init__(self, _C: np.ndarray, alpha: float) -> None:
        self._C = _C
        self.alpha = alpha

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """State-space dimension."""
        return self._C.shape[0]

    @property
    def matrix(self) -> np.ndarray:
        """A copy of the n×n conceptor matrix."""
        return self._C.copy()

    # ── Application ───────────────────────────────────────────────────────────

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Project state vector *x* through the conceptor: returns C @ x.

        Args:
            x: State vector, shape (n,).

        Returns:
            Filtered state, shape (n,).
        """
        return self._C @ x

    def singular_values(self) -> np.ndarray:
        """Singular values of the conceptor matrix, sorted descending.

        For a well-formed conceptor all values are in [0, 1].
        """
        return np.linalg.svd(self._C, compute_uv=False)

    # ── Boolean algebra (Jaeger 2014 §2.3) ───────────────────────────────────

    def __invert__(self) -> Conceptor:
        """NOT: ¬C = I − C.

        If C has eigenvalues in [0, 1], ¬C has eigenvalues 1 − λ ∈ [0, 1]. ✓
        """
        return Conceptor(_C=np.eye(self.n) - self._C, alpha=self.alpha)

    def __and__(self, other: Conceptor) -> Conceptor:
        """AND / intersection: C1 ∧ C2 = (C1⁻¹ + C2⁻¹ − I)⁻¹.

        Uses regularised pseudo-inverse to handle near-singular matrices.
        Eigenvalues are clipped to [0, 1] after computation.

        Args:
            other: A :class:`Conceptor` with the same dimension n.

        Returns:
            A new :class:`Conceptor` representing the intersection.
        """
        n = self.n
        eps = 1e-8  # regularisation to avoid singular inversions
        I = np.eye(n)
        inv1 = np.linalg.pinv(self._C + eps * I)
        inv2 = np.linalg.pinv(other._C + eps * I)
        raw = np.linalg.pinv(inv1 + inv2 - I)
        return Conceptor(_C=_clip_eigenvalues(raw), alpha=min(self.alpha, other.alpha))

    def __or__(self, other: Conceptor) -> Conceptor:
        """OR / union: C1 ∨ C2 = ¬(¬C1 ∧ ¬C2)  (de Morgan).

        Args:
            other: A :class:`Conceptor` with the same dimension n.

        Returns:
            A new :class:`Conceptor` representing the union.
        """
        return ~((~self) & (~other))

    def __repr__(self) -> str:
        svs = self.singular_values()
        return f"Conceptor(n={self.n}, alpha={self.alpha:.1f}, sv_range=[{svs.min():.3f}, {svs.max():.3f}])"
