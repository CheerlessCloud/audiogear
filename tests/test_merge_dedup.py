"""merge_shards must drop duplicate ids across shard files.

Rows are assigned to shards by ``row_index % world_size``, so changing
``executor.tasks`` between an interrupted run and its rerun re-partitions the
rows — old and new shard files then overlap, and a naive concat would put the
same clip into extended_metadata.csv twice.
"""

import csv
import importlib.util
import pathlib


def _load_run_batch(monkeypatch, data_root):
    monkeypatch.setenv("AUDIOGEAR_DATA_DIR", str(data_root))  # checked at import time
    path = pathlib.Path(__file__).parents[1] / "examples" / "run_batch.py"
    spec = importlib.util.spec_from_file_location("run_batch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text"], delimiter="|")
        w.writeheader()
        w.writerows(rows)


def test_merge_shards_drops_duplicate_ids(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    (data_root / "ds").mkdir(parents=True)
    rb = _load_run_batch(monkeypatch, data_root)
    monkeypatch.setattr(rb, "REPO", str(tmp_path))

    _write_shard(tmp_path / "outputs" / "ds" / "ext_00000.csv",
                 [{"id": "a", "text": "1"}, {"id": "b", "text": "2"}])
    _write_shard(tmp_path / "outputs" / "ds" / "ext_00001.csv",
                 [{"id": "b", "text": "2-dup"}, {"id": "c", "text": "3"}])

    n = rb.merge_shards("ds")
    assert n == 3
    with open(data_root / "ds" / "extended_metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert rows[1]["text"] == "2", "first occurrence wins"
    assert "duplicate id" in capsys.readouterr().out
