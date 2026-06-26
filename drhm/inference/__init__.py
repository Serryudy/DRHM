"""Volitional engine — Active Inference at moment 8 (CLAUDE.md §4, Phase 3/4).

Two modules work together:

  active_inference.py — GenerativeModel: EMA belief over X_t hypervectors;
                         surprise(x_t) = 1 − cosine(x_t, μ) ∈ [0, 2]
  javana.py           — JavanaDeterminer: EFE-based javana CittaType selection
                         at moment 8 (votthapana); replaces the M3 stub DeterminerFn.

Integration point: pass a ``JavanaDeterminer`` instance as the ``determiner``
argument to ``CittaVithi`` to activate M5 cognition.
"""
