import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from birdcall.eval.metrics import bootstrap_ci, macro_precision_at_k, precision_at_k


def test_precision_at_k_all_correct():
    assert precision_at_k(["A", "A", "A"], "A", 3) == 1.0


def test_precision_at_k_partial():
    assert precision_at_k(["A", "B", "A"], "A", 3) == 2 / 3


def test_precision_at_k_empty():
    assert precision_at_k([], "A", 5) == 0.0


def test_precision_at_k_respects_k():
    assert precision_at_k(["A", "B", "B", "B"], "A", 1) == 1.0


def test_macro_precision_at_k():
    queries = [
        {"true_species": "A", "retrieved_species": ["A", "A"]},
        {"true_species": "B", "retrieved_species": ["A", "B"]},
    ]
    result = macro_precision_at_k(queries, k=2)
    assert result["n_species"] == 2
    assert 0.0 <= result["macro_precision"] <= 1.0


def test_bootstrap_ci_bounds_constant_values():
    lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert lo == 1.0 and hi == 1.0


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")
