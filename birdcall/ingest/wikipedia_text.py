"""Fetch a short field-guide-style summary per species from Wikipedia (CC-BY-SA licensed) via
the REST summary API — no scraping/parsing needed.
"""

from pathlib import Path

import requests

from ..manifest import Item, Modality

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def fetch_species_text(species: str, out_dir) -> Item:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = species.replace(" ", "_")
    resp = requests.get(WIKI_API.format(title=title), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    extract = data.get("extract", "")
    if not extract:
        raise ValueError(f"No Wikipedia summary found for '{species}'")

    txt_path = out_dir / f"{title}.txt"
    txt_path.write_text(extract, encoding="utf-8")
    return Item(
        item_id=f"wiki_{title}",
        species=species,
        modality=Modality.TEXT,
        source="wikipedia",
        source_id=title,
        license="CC-BY-SA-4.0",
        url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        local_path=str(txt_path),
    )
