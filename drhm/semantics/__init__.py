"""Semantic grounding layer — the unit "X" (CLAUDE.md §4, Phase 3).

Three modules collaborate to turn raw SNN features into grounded, composable concepts:

  vsa.py              — bipolar {-1,+1} hypervectors; bind/bundle/permute; X formula
  conceptual_space.py — quality dimensions with Voronoi prototype regions
  conceptors.py       — C = R(R + α⁻²I)⁻¹; Boolean AND/OR/NOT over concept subspaces

Import from the sub-modules directly; this file is the package marker only.
"""
