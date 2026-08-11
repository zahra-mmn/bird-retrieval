"""Contrastive fine-tuning of the projection head: pulls same-species items from different
modalities together and pushes other species apart, via a multi-positive InfoNCE loss over
species-stratified batches (so every batch actually contains real positive pairs — random
batching over a ~300-450 item dataset would frequently produce zero same-species pairs).
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..manifest import Manifest, Split
from .projection_head import ProjectionHead

# Below this many training items for a species, its contribution to the contrastive loss is
# close to noise — surfaced as a warning rather than silently trained on.
MIN_ITEMS_FOR_TRAINING = 4


class SpeciesBatchSampler:
    """Samples `species_per_batch` species and up to `items_per_species` items each, across
    modalities, so every batch has real positive pairs for the contrastive loss.
    """

    def __init__(self, ids, meta, species_per_batch: int = 8, items_per_species: int = 4,
                 seed: int = 42):
        self.by_species = defaultdict(list)
        for idx, m in enumerate(meta):
            self.by_species[m.species].append(idx)
        self.species = [s for s, idxs in self.by_species.items() if len(idxs) >= 2]
        self.species_per_batch = max(1, min(species_per_batch, len(self.species)))
        self.items_per_species = items_per_species
        self.rng = random.Random(seed)

    def __iter__(self):
        species = self.species[:]
        self.rng.shuffle(species)
        for i in range(0, len(species), self.species_per_batch):
            chosen = species[i:i + self.species_per_batch]
            batch = []
            for sp in chosen:
                idxs = self.by_species[sp][:]
                self.rng.shuffle(idxs)
                batch.extend(idxs[: self.items_per_species])
            if len(batch) >= 4:
                yield batch

    def __len__(self) -> int:
        return max(1, len(self.species) // self.species_per_batch)


def multi_positive_info_nce(z: torch.Tensor, species_ids: torch.Tensor,
                             temperature: float = 0.1) -> torch.Tensor:
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    same_species = species_ids.unsqueeze(0) == species_ids.unsqueeze(1)
    same_species.fill_diagonal_(False)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_counts = same_species.sum(dim=1).clamp(min=1)
    loss_per_anchor = -(log_prob * same_species).sum(dim=1) / pos_counts
    valid = same_species.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.zeros((), requires_grad=True)
    return loss_per_anchor[valid].mean()


def train(manifest_path, embeddings_dir, out_dir, epochs: int = 30, species_per_batch: int = 8,
          items_per_species: int = 4, lr: float = 1e-3, temperature: float = 0.1,
          seed: int = 42, device: str | None = None) -> list[dict]:
    from ..embed.run_embedding import load_embeddings  # local import: keeps eval-only callers from needing torch's full embed stack

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest.load(manifest_path)
    train_manifest = Manifest([it for it in manifest.items if it.split == Split.TRAIN])

    species_counts = defaultdict(int)
    for it in train_manifest.items:
        species_counts[it.species] += 1
    too_small = [s for s, c in species_counts.items() if c < MIN_ITEMS_FOR_TRAINING]
    if too_small:
        print(f"[train] warning: {len(too_small)} species have < {MIN_ITEMS_FOR_TRAINING} "
              f"training items and will contribute little/no signal: {too_small}")

    ids, vecs, meta = load_embeddings(train_manifest, embeddings_dir)
    if len(ids) == 0:
        raise RuntimeError("No embeddings found — run `embed` before `train`.")

    species_list = sorted({m.species for m in meta})
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    species_ids_all = torch.tensor([species_to_idx[m.species] for m in meta])
    vecs_t = torch.tensor(vecs, dtype=torch.float32)

    head = ProjectionHead(in_dim=vecs_t.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sampler = SpeciesBatchSampler(ids, meta, species_per_batch, items_per_species, seed)

    history = []
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for batch_idx in sampler:
            batch_idx_t = torch.tensor(batch_idx)
            x = vecs_t[batch_idx_t].to(device)
            sp = species_ids_all[batch_idx_t].to(device)
            z = head(x)
            loss = multi_positive_info_nce(z, sp, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"[train] epoch {epoch}/{epochs} loss={mean_loss:.4f}")

    torch.save(head.state_dict(), out_dir / "projection_head.pt")
    with (out_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "species_index.json").write_text(json.dumps(species_list, indent=2))

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([h["epoch"] for h in history], [h["loss"] for h in history])
        plt.xlabel("epoch")
        plt.ylabel("multi-positive InfoNCE loss")
        plt.title("Projection head training")
        plt.savefig(out_dir / "training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        pass

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--out-dir", default="artifacts/projection_head")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--species-per-batch", type=int, default=8)
    p.add_argument("--items-per-species", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(args.manifest, args.embeddings_dir, args.out_dir, args.epochs,
          args.species_per_batch, args.items_per_species, args.lr, args.temperature, args.seed)


if __name__ == "__main__":
    main()
