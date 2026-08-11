# Bird Call Cross-Modal Retrieval

Query with a bird call (audio), a photo, or a text description — retrieve matching results
across all three modalities. Built on public wildlife datasets (Xeno-canto, iNaturalist,
Wikipedia), with a fine-tuned retrieval layer on top of frozen [ImageBind](https://github.com/facebookresearch/ImageBind)
embeddings, a structured failure-mode analysis, and an LLM reasoning layer that explains matches
and asks clarifying questions for genuinely ambiguous (confusable-species) queries.

This repo is a scaffold: every pipeline stage (ingest → preprocess → embed → train → index →
evaluate → reason → serve) is implemented and unit-testable end-to-end, but the data collection
and model training steps are meant to be *run* by you (in Colab, per the original scope doc) —
see [Running the pipeline](#running-the-pipeline) below.

## What's different from the original scope doc

Building this out surfaced a handful of things the original plan glossed over. All are fixed
here, not just noted:

1. **`pip install imagebind` doesn't work** — there's no `imagebind` package on PyPI. ImageBind
   must be installed from GitHub. `pyproject.toml` declares it as a git dependency
   (`imagebind @ git+https://github.com/facebookresearch/ImageBind.git`); see
   [Environment setup](#environment-setup).
2. **Raw field recordings are mostly silence.** ImageBind's audio encoder expects a short (~2s)
   clip; naively truncating a Xeno-canto recording to the first 2 seconds often captures no call
   at all. [`birdcall/audio/preprocess.py`](birdcall/audio/preprocess.py) instead picks the
   highest-energy 2-second window and estimates SNR, so what gets embedded is (probabilistically)
   the actual call, not silence — and the SNR estimate feeds directly into the noise-robustness
   failure-mode test.
3. **Contrastive fine-tuning was underspecified.** `birdcall/model/train_projection.py`
   implements a concrete multi-positive InfoNCE loss over species-stratified batches
   (`SpeciesBatchSampler`), so every batch actually contains same-species/different-modality
   positive pairs — random batching over a ~300-450 item dataset frequently wouldn't.
4. **5-10 items/species is thin for evaluation.** `Manifest.assign_splits()` emits an explicit
   warning for any species below a minimum item count instead of silently producing a
   precision@k number with no statistical weight behind it, and `eval/metrics.py` provides
   `bootstrap_ci()` so reported numbers come with a confidence interval, not a bare point
   estimate.
5. **Licensing was a checklist item, not code.** `birdcall/ingest/inaturalist.py` filters to a
   CC-license allowlist at download time — unlicensed photos never make it into the dataset in
   the first place.
6. **FAISS is overkill at this scale**, and the code says so:
   [`birdcall/index/faiss_index.py`](birdcall/index/faiss_index.py) deliberately uses a flat
   (brute-force, exact) index rather than IVF/HNSW — worth keeping for the portfolio signal, but
   honestly documented as unnecessary at ~300-450 items.
7. **LLM reasoning layer targets OpenAI**, not the Claude API the original doc assumed, since
   that's the key available for this build. The interface
   (`birdcall/reasoning/llm_reasoning.py`) is provider-agnostic — swapping `_call_llm` to use the
   `anthropic` SDK instead is a small, isolated change. Responses are cached to disk
   (`birdcall/reasoning/cache.py`) so iterating on the UI doesn't re-bill the API.
8. **Confusable-species pairs** (the doc's "bonus for interest") are seeded up front in
   `birdcall/species.py` rather than left as a stretch goal, since they're what the reasoning
   layer's ambiguity-handling and the failure-mode harness actually exercise.

## Architecture

```
Text/Image/Audio -> ImageBind (frozen) -> raw 1024-dim embedding
                                              |
                                              v
                              Projection Head (small MLP, trained
                              via multi-positive InfoNCE on species labels)
                                              |
                                              v
                        FAISS flat index (tagged: species/modality/source/license)
                                              |
                                              v
                                   Top-K candidate results
                                              |
                                              v
                     LLM Reasoning Layer (explains matches, flags low
                     confidence, asks a clarifying question if ambiguous)
                                              |
                                              v
                                    Streamlit chat-style UI
```

## Repository layout

```
birdcall/
  manifest.py             dataset manifest: item -> species/modality/source/license/split
  species.py               starter species list + seed confusable-pairs list
  audio/preprocess.py       call-isolation (highest-energy window + SNR estimate)
  ingest/                   xeno-canto / iNaturalist / Wikipedia downloaders
  embed/                    ImageBind wrapper + batch embedding orchestrator
  model/                    projection head + contrastive training
  index/faiss_index.py       flat FAISS index wrapper
  eval/                     precision@k + bootstrap CI + failure-mode harness
  reasoning/                 LLM reasoning layer + disk cache
  app/streamlit_app.py       demo UI
scripts/run_pipeline.py     CLI orchestrator (ingest / embed / train / index)
tests/                      dependency-light unit tests (metrics, manifest)
```

## Environment setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

### In Colab (data collection, embedding, training — Weeks 1-5)

```python
!pip install uv
!uv pip install --system -e .
```

(`-e .` installs everything declared in `pyproject.toml`, including ImageBind from GitHub —
this replaces the original doc's `pip install imagebind`, which targets a package that doesn't
exist.) ImageBind's git install can take a few minutes and needs `git` available in the Colab
image (it is, by default).

### Locally, for the Streamlit demo app (Week 6)

```powershell
uv sync
uv run streamlit run birdcall/app/streamlit_app.py
```

`uv sync` generates/reads `uv.lock` — commit it so a fresh clone reproduces the exact
environment with `uv sync` alone.

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` before running anything that touches the
reasoning layer.

## Running the pipeline

```powershell
uv run python scripts/run_pipeline.py ingest --data-dir data/raw --per-species 8
uv run python scripts/run_pipeline.py embed  --manifest data/raw/manifest.jsonl --out-dir data/embeddings
uv run python scripts/run_pipeline.py train  --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings

# build BOTH indices — this is what makes the Week 4 frozen-vs-fine-tuned comparison possible
uv run python scripts/run_pipeline.py index --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings \
    --out-dir artifacts/index/baseline
uv run python scripts/run_pipeline.py index --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings \
    --out-dir artifacts/index/finetuned --projection-head artifacts/projection_head/projection_head.pt

uv run streamlit run birdcall/app/streamlit_app.py
```

`ingest` downloads audio/images/text for every species in `birdcall/species.py`
(`DEFAULT_SPECIES` — edit this list first), runs audio through call-isolation, filters images to
CC-licensed ones, and writes `manifest.jsonl` with train/val/test splits assigned. Review the
printed warnings for any species with too few items before trusting later evaluation numbers.

`index` builds a **frozen-baseline** index (raw 1024-dim ImageBind space) when
`--projection-head` is omitted, and a **fine-tuned** index (256-dim projected space) when it's
given. The Streamlit app queries the fine-tuned index by default (`INDEX_DIR` in
`streamlit_app.py`) since it also projects the query through the trained head — the two spaces
aren't interchangeable, so don't point the app at the baseline index.

## Running the heavy stages in Colab

Ingestion, embedding, and training need a GPU and take a while — the intended split is: run
those stages in Colab (free T4), then pull the small result artifacts back locally via git to
run the Streamlit app. No Google Drive involved — Colab's local disk is ephemeral, so the repo
itself (via `git commit`/`git push`) is what carries results between sessions and back to your
machine, not a mounted drive.

Bulk source media (`data/raw/audio/`, `data/raw/images/`) stays gitignored — regenerate it with
`ingest` rather than committing it. Everything else (`manifest.jsonl`, cached `.npy` embeddings,
`artifacts/`) is small and deliberately tracked.

1. **In a new Colab notebook**, select a GPU runtime (Runtime → Change runtime type → T4 GPU),
   then clone + install:

   ```python
   %cd /content
   !rm -rf bird-retrieval
   !git clone https://github.com/zahra-mmn/bird-retrieval.git
   %cd /content/bird-retrieval
   !pip install uv
   !uv pip install --system -e .
   ```

2. **Set up git identity + push credentials**, so results can go back to GitHub from Colab.
   Generate a fine-grained Personal Access Token on the `zahra-mmn` account (Settings →
   Developer settings → Personal access tokens) scoped to this repo with **Contents:
   read/write**, then:

   ```python
   !git config user.email "you@example.com"
   !git config user.name "zahra-mmn"

   import getpass
   token = getpass.getpass("GitHub token: ")  # prompts, doesn't echo/print the value
   !git remote set-url origin https://{token}@github.com/zahra-mmn/bird-retrieval.git
   ```

3. **Run the pipeline stages directly against the repo's own `data/`/`artifacts/` folders**
   (no separate `DATA_DIR` needed):

   ```python
   !uv run python scripts/run_pipeline.py ingest --data-dir data/raw --species "American Crow,House Finch,Song Sparrow" --per-species 5
   !uv run python scripts/run_pipeline.py embed  --manifest data/raw/manifest.jsonl --out-dir data/embeddings
   !uv run python scripts/run_pipeline.py train  --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings --out-dir artifacts/projection_head
   !uv run python scripts/run_pipeline.py index  --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings --out-dir artifacts/index/baseline
   !uv run python scripts/run_pipeline.py index  --manifest data/raw/manifest.jsonl --embeddings-dir data/embeddings --out-dir artifacts/index/finetuned --projection-head artifacts/projection_head/projection_head.pt
   ```

4. **Commit and push the results** (git respects `.gitignore`, so this only picks up
   `manifest.jsonl`, the cached embeddings, and `artifacts/` — not the raw media):

   ```python
   !git add data/raw/manifest.jsonl data/raw/text data/embeddings artifacts
   !git commit -m "Colab run: ingest + embed + train + index"
   !git push
   ```

5. **Back on your local machine**, just pull:

   ```powershell
   git pull
   ```

   Then run the Streamlit app locally as usual — it reads `artifacts/index/finetuned` and
   `artifacts/projection_head/projection_head.pt`, both now present from the pull.

## Dataset licensing

- **Xeno-canto**: license recorded per-recording from the API's `lic` field into the manifest.
- **iNaturalist**: only `cc0` / `cc-by` / `cc-by-nc` / `cc-by-sa` / `cc-by-nc-sa` photos are
  downloaded; license recorded per-item.
- **Wikipedia**: text is CC-BY-SA-4.0.

Before using anything from `data/` in a public demo, spot-check the manifest's `license` field —
some CC variants (e.g. NC) restrict commercial use.

## Testing

`tests/` only exercises pure-Python logic (`birdcall/eval/metrics.py`,
`birdcall/manifest.py`) — no torch/faiss/imagebind required:

```powershell
uv run pytest tests/
# or, with no dependencies installed at all:
python tests/test_metrics.py
python tests/test_manifest.py
```

## Known limitations

- Small per-species sample sizes (per the doc's own 5-10 items/species recommendation) mean
  precision@k numbers need the bootstrap CI from `eval/metrics.py`, not bare point estimates —
  and species below `MIN_ITEMS_FOR_TRAINING` (4) contribute little to the contrastive loss.
- ImageBind's audio encoder is general-purpose (AudioSet-trained), not bioacoustics-specific;
  call-isolation preprocessing helps but doesn't fully compensate — this is exactly what the
  failure-mode harness's noise-robustness test is for.
- The flat FAISS index is exact but O(n) per query; fine at hundreds of items, would need
  revisiting (IVF/HNSW) if the species list grows into the thousands of items.
