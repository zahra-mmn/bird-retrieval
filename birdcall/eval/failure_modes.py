"""Structured failure-mode analysis harness covering the categories the scope doc calls for:
confusable-species confusion, background-noise robustness, and call-vs-text-description query
behavior. Produces a markdown + JSON report with a `hypothesis` field left as a template — the
harness structures the analysis, but *why* a given failure happens still needs a human looking
at the actual retrieved items, per the doc's "document ... with a hypothesis" requirement.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FailureCase:
    category: str
    query_item_id: str
    query_species: str
    top_result_species: str
    top_result_score: float
    example_detail: str
    hypothesis: str = "TODO: fill in after reviewing the retrieved items"


class FailureModeHarness:
    def __init__(self, index, k: int = 5):
        self.index = index
        self.k = k
        self.cases: list[FailureCase] = []

    def test_confusable_pairs(self, pairs: list[dict], query_fn) -> None:
        """`query_fn(species, modality) -> (item_id, vector) | None` — caller supplies how to
        pick a representative query item for a species/modality from the held-out set.
        """
        for pair in pairs:
            for species, other in ((pair["species_a"], pair["species_b"]),
                                    (pair["species_b"], pair["species_a"])):
                modality = "audio" if pair["type"] == "acoustic" else "image"
                picked = query_fn(species, modality)
                if picked is None:
                    continue
                item_id, vec = picked
                results = self.index.search(vec, k=self.k)
                top = results[0] if results else None
                if top and top["species"] != species:
                    self.cases.append(FailureCase(
                        category=f"confusable_pair_{pair['type']}",
                        query_item_id=item_id,
                        query_species=species,
                        top_result_species=top["species"],
                        top_result_score=top["score"],
                        example_detail=pair.get("note", ""),
                    ))

    def test_noise_robustness(self, noisy_items, clean_items, query_fn) -> None:
        """`query_fn(item) -> (item_id, vector) | None`."""
        for label, items in (("noisy", noisy_items), ("clean", clean_items)):
            for it in items:
                picked = query_fn(it)
                if picked is None:
                    continue
                item_id, vec = picked
                results = self.index.search(vec, k=self.k)
                top = results[0] if results else None
                if not top or top["species"] != it.species:
                    self.cases.append(FailureCase(
                        category=f"noise_robustness_{label}",
                        query_item_id=item_id,
                        query_species=it.species,
                        top_result_species=top["species"] if top else "none",
                        top_result_score=top["score"] if top else 0.0,
                        example_detail=f"snr_estimate={it.extra.get('snr_estimate')}",
                    ))

    def test_call_vs_text_query(self, species_list: list[str], query_fn) -> None:
        """`query_fn(species, modality) -> (item_id, vector) | None`."""
        for species in species_list:
            audio_pick = query_fn(species, "audio")
            text_pick = query_fn(species, "text")
            for label, picked in (("call_query", audio_pick), ("text_query", text_pick)):
                if picked is None:
                    continue
                item_id, vec = picked
                results = self.index.search(vec, k=self.k)
                top = results[0] if results else None
                if not top or top["species"] != species:
                    self.cases.append(FailureCase(
                        category=f"modality_asymmetry_{label}",
                        query_item_id=item_id,
                        query_species=species,
                        top_result_species=top["species"] if top else "none",
                        top_result_score=top["score"] if top else 0.0,
                        example_detail="query-by-call vs query-by-text-description comparison",
                    ))

    def report(self, out_path) -> None:
        out_path = Path(out_path)
        by_category: dict = {}
        for c in self.cases:
            by_category.setdefault(c.category, []).append(c)

        lines = ["# Failure-Mode Analysis\n"]
        for category, cases in by_category.items():
            lines.append(f"## {category} ({len(cases)} cases)\n")
            for c in cases:
                lines.append(
                    f"- `{c.query_item_id}` ({c.query_species}) -> top result "
                    f"**{c.top_result_species}** (score={c.top_result_score:.3f}). "
                    f"{c.example_detail}\n  - Hypothesis: {c.hypothesis}"
                )
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        out_path.with_suffix(".json").write_text(
            json.dumps([vars(c) for c in self.cases], indent=2), encoding="utf-8"
        )
