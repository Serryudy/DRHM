"""Vector-Symbolic Architecture — hyperdimensional computing engine (CLAUDE.md §4, M4).

ADR-VSA-001 (architecture decision)
────────────────────────────────────
Framework choice: pure numpy, no torchhd.
Reason: edge-deployability and auditable footprint (Transcendence §8).  torchhd pulls in
the full PyTorch stack; for bipolar {-1,+1} algebra numpy is sufficient and roughly as
fast for D=10 000.  If torch becomes a hard dep for other reasons, migrate then.

Bipolar {-1, +1} properties exploited here
───────────────────────────────────────────
  ||v||² = D  for every bipolar vector  ⟹  cosine(a, b) = dot(a, b) / D  (no sqrt needed)
  bind(a, b) = a ⊗ b  is its own inverse:  bind(bind(a,b), b) = a
  bundle truncates noisy membership: each element votes ±1; majority wins

Reference: Kanerva 2009 "Hyperdimensional computing"; Plate 1995 (holographic reduced
           representations); Frady et al. 2021 (resonator networks & VSA overview).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drhm import config


@dataclass
class VSA:
    """Hyperdimensional computing engine operating over bipolar {-1, +1} vectors.

    One ``VSA`` instance owns a fixed random permutation and an RNG.  A single
    shared instance should be used throughout the agent so the permutation is
    consistent across modules.

    Args:
        D:   Hypervector dimensionality.  Defaults to ``config.VSA_D`` (10 000).
        rng: NumPy random generator.  Defaults to ``np.random.default_rng(config.SEED)``.

    Attributes:
        D: Dimensionality of every vector produced by this instance.
    """

    D: int = field(default_factory=lambda: config.VSA_D)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(config.SEED))
    # Fixed permutation — set in __post_init__; must not change after construction.
    _perm: np.ndarray = field(init=False, repr=False)
    _inv_perm: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._perm = self.rng.permutation(self.D).astype(np.intp)
        self._inv_perm = np.argsort(self._perm).astype(np.intp)

    # ── Generation ────────────────────────────────────────────────────────────

    def random(self) -> np.ndarray:
        """Generate a random bipolar hypervector of shape (D,), dtype int8."""
        return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=self.D)

    # ── Core operations ───────────────────────────────────────────────────────

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """⊗ Hadamard product — losslessly invertible: bind(bind(a, b), b) == a.

        The result is orthogonal (≈0 cosine) to both operands when a and b are
        independent random vectors.  Used to associate (role, filler) pairs.

        Args:
            a: Bipolar hypervector, shape (D,).
            b: Bipolar hypervector, shape (D,).

        Returns:
            Element-wise product, shape (D,), dtype int8, bipolar {-1, +1}.
        """
        return (a * b).astype(np.int8)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """⊕ Superposition — result is similar to every constituent.

        Each element is the sign of the element-wise sum.  Ties (when len(vectors)
        is even) are broken uniformly at random.

        Args:
            vectors: Non-empty list of bipolar hypervectors, each shape (D,).

        Returns:
            Bipolar hypervector, shape (D,), dtype int8.

        Raises:
            ValueError: If *vectors* is empty.
        """
        if not vectors:
            raise ValueError("bundle requires at least one vector")
        s = np.sum(np.stack(vectors).astype(np.int32), axis=0)
        result = np.sign(s).astype(np.int8)
        ties = result == 0
        if ties.any():
            result[ties] = self.rng.choice(
                np.array([-1, 1], dtype=np.int8), size=int(ties.sum())
            )
        return result

    def permute(self, v: np.ndarray) -> np.ndarray:
        """Π Apply the fixed random permutation — makes sequence order distinguishable.

        bind(A, permute(B)) encodes "A in context of B at position+1", which differs
        from bind(B, permute(A)).  This breaks the commutativity of bind and lets VSA
        represent ordered sequences.

        Args:
            v: Bipolar hypervector, shape (D,).

        Returns:
            Permuted copy, shape (D,), dtype int8.
        """
        return v[self._perm]

    def inv_permute(self, v: np.ndarray) -> np.ndarray:
        """Π⁻¹ Reverse the permutation applied by :meth:`permute`."""
        return v[self._inv_perm]

    # ── Similarity ────────────────────────────────────────────────────────────

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity in [-1, +1].

        For bipolar {-1,+1} vectors ||v||² = D, so cosine = dot(a, b) / D.
        Two independent random vectors give cosine ≈ 0 (std ≈ 1/√D).
        """
        return float(np.dot(a.astype(np.int32), b.astype(np.int32))) / self.D

    # ── Composite encoding (the unit "X") ─────────────────────────────────────

    def encode_x(
        self,
        context: np.ndarray,
        structure: np.ndarray,
        features: list[np.ndarray],
    ) -> np.ndarray:
        """Compute the unit X_t = (C ⊗ S) ⊕ Σᵢ(C ⊗ Fᵢ) — the grounded percept.

        Every feature and the structural frame are stamped with the context vector,
        making X_t specific to the situation it was perceived in.  The same "red"
        feature produces a different X_t in a "traffic-light" context vs. an "apple"
        context (CLAUDE.md §3, docs/Architecture of X.md).

        Args:
            context:   Context hypervector C — encodes the situational frame.
            structure: Structure hypervector S — the abstract schema / image schema.
            features:  List of feature hypervectors Fᵢ — perceptual content.

        Returns:
            Composite bipolar hypervector X_t, shape (D,).
        """
        structure_term = self.bind(context, structure)
        if not features:
            return structure_term
        feature_bundle = self.bundle([self.bind(context, f) for f in features])
        return self.bundle([structure_term, feature_bundle])
