"""Dataset manifest: tracks every ingested item with species, modality, source, license, and
split, so the exact dataset behind any run can be reproduced or audited later.
"""

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Modality(str, Enum):
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class Item:
    item_id: str
    species: str
    modality: Modality
    source: str
    source_id: str
    license: str
    url: str
    local_path: str
    split: Optional[Split] = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modality"] = self.modality.value
        d["split"] = self.split.value if self.split else None
        return d


class Manifest:
    def __init__(self, items: Optional[list] = None):
        self.items: list[Item] = items or []

    def add(self, item: Item) -> None:
        self.items.append(item)

    def by_species(self) -> dict:
        grouped = defaultdict(list)
        for it in self.items:
            grouped[it.species].append(it)
        return grouped

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it.to_dict()) + "\n")

    @classmethod
    def load(cls, path) -> "Manifest":
        items = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["modality"] = Modality(d["modality"])
                d["split"] = Split(d["split"]) if d.get("split") else None
                items.append(Item(**d))
        return cls(items)

    def version_hash(self) -> str:
        ids = sorted(it.item_id for it in self.items)
        return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:12]

    def assign_splits(
        self,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        min_items_per_species: int = 5,
        seed: int = 42,
    ) -> list[str]:
        """Stratified split per species/modality. Returns human-readable warnings for species
        too small to produce a statistically meaningful held-out precision number — surfacing
        this instead of silently reporting a misleading metric.
        """
        rng = random.Random(seed)
        warnings = []
        for species, its in self.by_species().items():
            if len(its) < min_items_per_species:
                warnings.append(
                    f"{species}: only {len(its)} items total — split sizes will be too small "
                    f"for a statistically meaningful test precision number; treat results for "
                    f"this species as anecdotal, not a metric."
                )
            by_mod = defaultdict(list)
            for it in its:
                by_mod[it.modality].append(it)
            for _mod, mod_items in by_mod.items():
                rng.shuffle(mod_items)
                n = len(mod_items)
                n_test = max(1, round(n * test_frac)) if n >= 3 else 0
                n_val = max(1, round(n * val_frac)) if n - n_test >= 3 else 0
                for it in mod_items[:n_test]:
                    it.split = Split.TEST
                for it in mod_items[n_test : n_test + n_val]:
                    it.split = Split.VAL
                for it in mod_items[n_test + n_val :]:
                    it.split = Split.TRAIN
        return warnings
