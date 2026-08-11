"""Embed every item in a manifest with ImageBind and cache raw 1024-dim vectors to disk, keyed
by item_id, so training/indexing/eval never have to re-run the (slow) encoder.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..manifest import Manifest
from .imagebind_encoder import ImageBindEncoder


def embed_manifest(manifest_path, out_dir, batch_size: int = 8, device: str | None = None) -> None:
    manifest = Manifest.load(manifest_path)
    encoder = ImageBindEncoder(device=device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_modality: dict = {"audio": [], "image": [], "text": []}
    for it in manifest.items:
        by_modality[it.modality.value].append(it)

    for modality, items in by_modality.items():
        for i in tqdm(range(0, len(items), batch_size), desc=f"embedding {modality}"):
            batch = items[i:i + batch_size]
            payload = [
                _read_text(it.local_path) if modality == "text" else it.local_path
                for it in batch
            ]
            vecs = encoder.encode_modality(modality, payload).cpu().numpy()
            for it, vec in zip(batch, vecs):
                np.save(out_dir / f"{it.item_id}.npy", vec)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_embeddings(manifest: Manifest, embeddings_dir):
    embeddings_dir = Path(embeddings_dir)
    ids, vecs, meta = [], [], []
    for it in manifest.items:
        p = embeddings_dir / f"{it.item_id}.npy"
        if not p.exists():
            continue
        ids.append(it.item_id)
        vecs.append(np.load(p))
        meta.append(it)
    return ids, np.stack(vecs) if vecs else np.zeros((0, 1024)), meta
