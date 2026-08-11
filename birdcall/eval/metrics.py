"""Retrieval quality metrics. Pure Python/stdlib — no torch/faiss/imagebind — so these are
unit-testable without any of the heavy ML dependencies installed.
"""

from collections import defaultdict


def precision_at_k(retrieved_species: list[str], true_species: str, k: int) -> float:
    top_k = retrieved_species[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if s == true_species)
    return hits / len(top_k)


def macro_precision_at_k(queries: list[dict], k: int) -> dict:
    """`queries`: list of {"true_species": str, "retrieved_species": list[str]}."""
    per_species = defaultdict(list)
    for q in queries:
        p = precision_at_k(q["retrieved_species"], q["true_species"], k)
        per_species[q["true_species"]].append(p)
    per_species_mean = {sp: sum(vals) / len(vals) for sp, vals in per_species.items()}
    macro = sum(per_species_mean.values()) / len(per_species_mean) if per_species_mean else 0.0
    return {"macro_precision": macro, "per_species": per_species_mean, "n_species": len(per_species_mean)}


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95,
                  seed: int = 42) -> tuple:
    """95% CI via bootstrap resampling. With only 5-10 items per species (see the scope doc's
    own dataset plan), a bare precision number is easy to over-read — pair it with this.
    """
    import random
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = min(int((1 + ci) / 2 * n_resamples), n_resamples - 1)
    return (means[lo_idx], means[hi_idx])
