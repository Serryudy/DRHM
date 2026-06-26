"""Conceptual Spaces — grounded semantic regions (CLAUDE.md §4, M4).

Theory (Gärdenfors 2000 *Conceptual Spaces*):
  A conceptual space is a metric space of *quality dimensions* (perceptual axes).
  Categories are *convex* regions; each region is anchored by a *prototype* — the
  "best example" of that category.  Membership is decided by nearest-prototype
  (Voronoi) lookup, which automatically gives convex regions in Euclidean space.

In DRHM, conceptual spaces form the bridge between raw SNN activations (Phase 2)
and VSA hypervectors (Phase 3):
  SNN features → quality-space coordinates → nearest prototype → hypervector

Each prototype owns a unique random hypervector generated at registration time.
The mapping is therefore:
  identical quality-space coordinates → identical hypervector (deterministic lookup)
  nearby coordinates (same Voronoi cell) → same prototype → same hypervector

Reference: Gärdenfors 2000, §2; docs/Architecture of X.md §3 (sensory grounding).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drhm.semantics.vsa import VSA


# ── Prototype ─────────────────────────────────────────────────────────────────

@dataclass
class Prototype:
    """A named exemplar in a conceptual space.

    Attributes:
        name:        Human-readable category label (e.g. ``"C4"``, ``"red"``).
        coords:      Position in quality space, shape (n_dims,).
        hypervector: Pre-generated VSA representation, shape (D,), bipolar {-1, +1}.
    """

    name: str
    coords: np.ndarray
    hypervector: np.ndarray


# ── ConceptualSpace ───────────────────────────────────────────────────────────

@dataclass
class ConceptualSpace:
    """A set of quality dimensions with Voronoi-partitioned prototype regions.

    Quality dimensions are numerical axes (e.g. frequency, hue, spatial-x).
    Categories are the Voronoi cells of the registered prototypes; every Voronoi
    cell in Euclidean space is convex, satisfying Gärdenfors' convexity requirement.

    The space is populated by calling :meth:`add_prototype`.  Once populated,
    :meth:`encode` maps a quality-space point to its category's hypervector, and
    :meth:`decode` maps a hypervector back to the nearest prototype name.

    Args:
        n_dims: Number of quality dimensions.
        vsa:    :class:`~drhm.semantics.vsa.VSA` instance used to generate
                prototype hypervectors.
    """

    n_dims: int
    vsa: VSA
    _prototypes: list[Prototype] = field(default_factory=list, init=False, repr=False)

    # ── Prototype management ──────────────────────────────────────────────────

    def add_prototype(self, name: str, coords: np.ndarray) -> np.ndarray:
        """Register a named prototype at *coords* and return its hypervector.

        Each call to ``add_prototype`` generates a unique random hypervector for
        that category.  Two prototypes at different positions always get different
        hypervectors, even if they happen to have the same Voronoi cell later.

        Args:
            name:   Category label (must be unique within this space).
            coords: Quality-space coordinates, shape (n_dims,).

        Returns:
            The prototype's hypervector, shape (D,), bipolar {-1, +1}.
        """
        hv = self.vsa.random()
        self._prototypes.append(Prototype(name=name, coords=coords.copy(), hypervector=hv))
        return hv

    @property
    def prototypes(self) -> list[Prototype]:
        """Ordered list of registered prototypes (read-only view)."""
        return list(self._prototypes)

    # ── Core operations ───────────────────────────────────────────────────────

    def nearest(self, coords: np.ndarray) -> Prototype:
        """Voronoi lookup: return the prototype closest to *coords*.

        Args:
            coords: Quality-space point, shape (n_dims,).

        Returns:
            The :class:`Prototype` whose position is nearest in Euclidean distance.

        Raises:
            ValueError: If no prototypes have been registered.
        """
        if not self._prototypes:
            raise ValueError("ConceptualSpace has no prototypes — call add_prototype() first")
        dists = np.array([np.linalg.norm(coords - p.coords) for p in self._prototypes])
        return self._prototypes[int(dists.argmin())]

    def encode(self, coords: np.ndarray) -> np.ndarray:
        """Map a quality-space point to the nearest prototype's hypervector.

        Args:
            coords: Quality-space coordinates, shape (n_dims,).

        Returns:
            Hypervector of the nearest prototype, shape (D,).
        """
        return self.nearest(coords).hypervector

    def decode(self, hv: np.ndarray) -> str:
        """Find the best-matching prototype name by cosine similarity.

        Args:
            hv: A hypervector, shape (D,).

        Returns:
            Name of the prototype with the highest cosine similarity to *hv*.

        Raises:
            ValueError: If no prototypes have been registered.
        """
        if not self._prototypes:
            raise ValueError("ConceptualSpace has no prototypes")
        best_name, best_sim = "", -2.0
        for p in self._prototypes:
            s = self.vsa.cosine(hv, p.hypervector)
            if s > best_sim:
                best_sim = s
                best_name = p.name
        return best_name

    def is_in_region(self, coords: np.ndarray, name: str) -> bool:
        """True iff *coords* falls within the Voronoi cell of prototype *name*.

        Args:
            coords: Quality-space coordinates, shape (n_dims,).
            name:   Target prototype name.

        Returns:
            ``True`` if the nearest prototype has the given name.
        """
        return self.nearest(coords).name == name

    # ── Geometry helpers (used in convexity tests) ────────────────────────────

    @staticmethod
    def midpoint(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Convex interpolation between two quality-space points.

        Returns ``(1 - alpha) * a + alpha * b``.  When both *a* and *b* belong to
        the same prototype region, the midpoint must also belong to it
        (Gärdenfors' convexity axiom, satisfied by Voronoi cells in Euclidean space).
        """
        return (1.0 - alpha) * a + alpha * b
