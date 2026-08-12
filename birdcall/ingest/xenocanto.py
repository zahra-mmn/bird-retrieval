"""Download bird-call recordings from Xeno-canto, run them through call-isolation preprocessing,
and record exact source IDs + licenses into the manifest.

Xeno-canto retired its v2 API in favor of v3, which requires a free API key (register at
xeno-canto.org with a verified email, then find your key on your account page) — set
XENO_CANTO_API_KEY in the environment or .env before calling this module.
"""

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from ..audio.preprocess import isolate_call
from ..manifest import Item, Modality

load_dotenv()

XC_API = "https://xeno-canto.org/api/3/recordings"


def search_recordings(species: str, max_results: int = 10, quality: str = "A") -> list[dict]:
    api_key = os.environ.get("XENO_CANTO_API_KEY")
    if not api_key:
        raise RuntimeError(
            "XENO_CANTO_API_KEY is not set. Xeno-canto's v3 API requires a free key — "
            "register at xeno-canto.org (verified email), grab the key from your account "
            "page, then set it in .env or `os.environ`."
        )
    query = f'en:"{species}" q:{quality}'
    resp = requests.get(XC_API, params={"query": query, "key": api_key}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Xeno-canto API error: {data.get('error')} — {data.get('message')}")
    return data.get("recordings", [])[:max_results]


def download_species_audio(species: str, out_dir, max_results: int = 10,
                            target_sr: int = 16000, target_duration: float = 2.0) -> list[Item]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = search_recordings(species, max_results)
    items = []
    for rec in recordings:
        rec_id = rec.get("id")
        file_url = rec.get("file")
        if not rec_id or not file_url:
            continue
        raw_path = out_dir / f"xc{rec_id}_raw.mp3"
        wav_path = out_dir / f"xc{rec_id}.wav"
        try:
            _download(file_url, raw_path)
            preprocess_info = isolate_call(raw_path, wav_path, target_sr=target_sr,
                                            target_duration=target_duration)
        except Exception as e:
            print(f"[xeno-canto] skipping {rec_id} ({species}): {e}")
            continue
        finally:
            raw_path.unlink(missing_ok=True)

        items.append(Item(
            item_id=f"xc_{rec_id}",
            species=species,
            modality=Modality.AUDIO,
            source="xeno-canto",
            source_id=str(rec_id),
            license=rec.get("lic", "unknown"),
            url=f"https://xeno-canto.org/{rec_id}",
            local_path=str(wav_path),
            extra={
                "quality": rec.get("q"),
                "recordist": rec.get("rec"),
                "snr_estimate": preprocess_info["snr_estimate"],
                "call_start_sec": preprocess_info["start_sec"],
            },
        ))
        time.sleep(1)  # be polite to the API
    return items


def _download(url: str, dest: Path) -> None:
    if url.startswith("//"):
        url = "https:" + url
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
