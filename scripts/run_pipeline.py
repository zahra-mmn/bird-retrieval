"""Orchestrates the full pipeline stage by stage. See README for the full week-by-week sequence.

Usage:
    uv run python scripts/run_pipeline.py ingest --data-dir data/raw --per-species 8
    uv run python scripts/run_pipeline.py embed  --manifest data/raw/manifest.jsonl --out-dir data/embeddings
    uv run python scripts/run_pipeline.py train  --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings
    uv run python scripts/run_pipeline.py index  --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings
"""

import argparse
from pathlib import Path

from birdcall.ingest import inaturalist, wikipedia_text, xenocanto
from birdcall.manifest import Manifest
from birdcall.species import DEFAULT_SPECIES


def cmd_ingest(args) -> None:
    species_list = [s.strip() for s in args.species.split(",")] if args.species else DEFAULT_SPECIES
    manifest = Manifest()
    for species in species_list:
        print(f"[ingest] {species}")
        try:
            manifest.items.extend(xenocanto.download_species_audio(
                species, Path(args.data_dir) / "audio", args.per_species))
        except Exception as e:
            print(f"  audio failed: {e}")
        try:
            manifest.items.extend(inaturalist.download_species_images(
                species, Path(args.data_dir) / "images", args.per_species))
        except Exception as e:
            print(f"  images failed: {e}")
        try:
            manifest.items.append(wikipedia_text.fetch_species_text(
                species, Path(args.data_dir) / "text"))
        except Exception as e:
            print(f"  text failed: {e}")

    warnings = manifest.assign_splits()
    for w in warnings:
        print(f"[ingest] warning: {w}")

    manifest_path = Path(args.data_dir) / "manifest.jsonl"
    manifest.save(manifest_path)
    print(f"[ingest] wrote {len(manifest.items)} items to {manifest_path} "
          f"(manifest version {manifest.version_hash()})")


def cmd_embed(args) -> None:
    from birdcall.embed.run_embedding import embed_manifest
    embed_manifest(args.manifest, args.out_dir)


def cmd_train(args) -> None:
    from birdcall.model.train_projection import train
    train(args.manifest, args.embeddings_dir, args.out_dir, epochs=args.epochs)


def cmd_index(args) -> None:
    from birdcall.embed.run_embedding import load_embeddings
    from birdcall.index.faiss_index import build_index_from_manifest
    manifest = Manifest.load(args.manifest)
    ids, vecs, meta = load_embeddings(manifest, args.embeddings_dir)

    if args.projection_head:
        # Index must live in the same space the app's query embeddings get projected into —
        # otherwise a 256-dim projected query hits a 1024-dim raw index and search breaks.
        import torch
        from birdcall.model.projection_head import ProjectionHead
        head = ProjectionHead(in_dim=vecs.shape[1])
        head.load_state_dict(torch.load(args.projection_head, map_location="cpu"))
        head.eval()
        with torch.no_grad():
            vecs = head(torch.tensor(vecs, dtype=torch.float32)).numpy()

    idx = build_index_from_manifest(ids, vecs, meta)
    idx.save(args.out_dir)
    space = "projected (fine-tuned)" if args.projection_head else "raw (frozen baseline)"
    print(f"[index] indexed {len(ids)} items in {space} space -> {args.out_dir}")


def main():
    p = argparse.ArgumentParser(description="Bird cross-modal retrieval pipeline")
    sub = p.add_subparsers(dest="stage", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--data-dir", default="data/raw")
    p_ingest.add_argument("--per-species", type=int, default=8)
    p_ingest.add_argument("--species", default=None,
                           help="Comma-separated species list to override birdcall/species.py's "
                                "DEFAULT_SPECIES — useful for a fast smoke test, e.g. "
                                "'American Crow,House Finch,Song Sparrow'")
    p_ingest.set_defaults(func=cmd_ingest)

    p_embed = sub.add_parser("embed")
    p_embed.add_argument("--manifest", default="data/raw/manifest.jsonl")
    p_embed.add_argument("--out-dir", default="data/embeddings")
    p_embed.set_defaults(func=cmd_embed)

    p_train = sub.add_parser("train")
    p_train.add_argument("--manifest", default="data/raw/manifest.jsonl")
    p_train.add_argument("--embeddings-dir", default="data/embeddings")
    p_train.add_argument("--out-dir", default="artifacts/projection_head")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.set_defaults(func=cmd_train)

    p_index = sub.add_parser("index")
    p_index.add_argument("--manifest", default="data/raw/manifest.jsonl")
    p_index.add_argument("--embeddings-dir", default="data/embeddings")
    p_index.add_argument("--out-dir", default="artifacts/index")
    p_index.add_argument("--projection-head", default=None,
                          help="Path to a trained projection_head.pt. Omit to build the frozen "
                               "baseline index (raw ImageBind space); pass it to build the "
                               "fine-tuned index (projected space) — build both for the "
                               "Week 4 baseline-vs-fine-tuned comparison.")
    p_index.set_defaults(func=cmd_index)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
