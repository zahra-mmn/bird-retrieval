"""FAISS index over projected embeddings.

Deliberately uses a flat (brute-force) `IndexFlatIP` index rather than IVF/HNSW: at ~300-450
items, an approximate index would be premature optimization that only adds recall risk, and
flat search is fast enough at this scale. Revisit only if the dataset grows to tens of
thousands of items.
"""

import json
from pathlib import Path

import faiss
import numpy as np


class BirdIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        vectors = _l2_normalize(vectors.astype("float32"))
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k: int = 10, modality_filter: str | None = None) -> list[dict]:
        query = _l2_normalize(query.astype("float32").reshape(1, -1))
        search_k = k if modality_filter is None else min(len(self.metadata), max(k * 5, k))
        search_k = max(1, search_k)
        scores, idxs = self.index.search(query, search_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if modality_filter and meta["modality"] != modality_filter:
                continue
            results.append({"score": float(score), **meta})
            if len(results) >= k:
                break
        return results

    def save(self, out_dir) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        (out_dir / "metadata.json").write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, out_dir) -> "BirdIndex":
        out_dir = Path(out_dir)
        index = faiss.read_index(str(out_dir / "index.faiss"))
        metadata = json.loads((out_dir / "metadata.json").read_text())
        obj = cls(dim=index.d)
        obj.index = index
        obj.metadata = metadata
        return obj


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)


def build_index_from_manifest(ids, vecs: np.ndarray, meta: list) -> BirdIndex:
    idx = BirdIndex(dim=vecs.shape[1])
    metadata = [
        {"item_id": i, "species": m.species, "modality": m.modality.value,
         "source": m.source, "local_path": m.local_path, "license": m.license}
        for i, m in zip(ids, meta)
    ]
    idx.add(vecs, metadata)
    return idx
