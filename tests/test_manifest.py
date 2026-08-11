import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from birdcall.manifest import Item, Manifest, Modality


def _make_item(i, species, modality):
    return Item(item_id=f"{species}_{modality.value}_{i}", species=species, modality=modality,
                source="test", source_id=str(i), license="cc0", url="", local_path="")


def test_save_load_roundtrip():
    m = Manifest([_make_item(0, "Robin", Modality.AUDIO)])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "manifest.jsonl"
        m.save(path)
        loaded = Manifest.load(path)
    assert len(loaded.items) == 1
    assert loaded.items[0].species == "Robin"
    assert loaded.items[0].modality == Modality.AUDIO


def test_assign_splits_warns_on_small_species():
    items = [_make_item(i, "Rare Bird", Modality.AUDIO) for i in range(2)]
    m = Manifest(items)
    warnings = m.assign_splits(min_items_per_species=5)
    assert any("Rare Bird" in w for w in warnings)


def test_assign_splits_covers_every_item():
    items = [_make_item(i, "Robin", Modality.AUDIO) for i in range(10)]
    m = Manifest(items)
    m.assign_splits()
    assert all(it.split is not None for it in m.items)


def test_version_hash_stable_regardless_of_order():
    a = Manifest([_make_item(0, "Robin", Modality.AUDIO), _make_item(1, "Wren", Modality.AUDIO)])
    b = Manifest([_make_item(1, "Wren", Modality.AUDIO), _make_item(0, "Robin", Modality.AUDIO)])
    assert a.version_hash() == b.version_hash()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all tests passed")
