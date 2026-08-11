"""Download research-grade, CC-licensed photos from iNaturalist for each species, recording
per-photo license into the manifest so unlicensed items never reach a public demo/repo.
"""

import time
from pathlib import Path

import requests

from ..manifest import Item, Modality

INAT_API = "https://api.inaturalist.org/v1/observations"
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-nc", "cc-by-sa", "cc-by-nc-sa"}


def search_photos(species: str, max_results: int = 10) -> list[dict]:
    params = {
        "taxon_name": species,
        "photos": "true",
        "quality_grade": "research",
        "license": ",".join(ALLOWED_LICENSES),
        "per_page": max_results,
        "order_by": "votes",
    }
    resp = requests.get(INAT_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def download_species_images(species: str, out_dir, max_results: int = 10) -> list[Item]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observations = search_photos(species, max_results)
    items = []
    for obs in observations:
        photos = obs.get("photos") or []
        if not photos:
            continue
        photo = photos[0]
        license_code = (photo.get("license_code") or "unknown").lower()
        if license_code not in ALLOWED_LICENSES:
            continue
        url = (photo.get("url") or "").replace("square", "medium")
        if not url:
            continue
        img_path = out_dir / f"inat{obs['id']}.jpg"
        try:
            _download(url, img_path)
        except Exception as e:
            print(f"[inaturalist] skipping {obs['id']} ({species}): {e}")
            continue
        items.append(Item(
            item_id=f"inat_{obs['id']}",
            species=species,
            modality=Modality.IMAGE,
            source="inaturalist",
            source_id=str(obs["id"]),
            license=license_code,
            url=obs.get("uri", url),
            local_path=str(img_path),
            extra={"observer": (obs.get("user") or {}).get("login")},
        ))
        time.sleep(1)
    return items


def _download(url: str, dest: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
