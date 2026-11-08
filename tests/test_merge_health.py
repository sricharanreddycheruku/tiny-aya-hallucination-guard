import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from merge_health import merge


def test_merge_drops_repeats_and_existing_rows(tmp_path):
    rows = [
        {"id": "a", "language": "en", "question": "What causes malaria?"},
        {"id": "b", "language": "en", "question": "what causes malaria"},
        {"id": "c", "language": "en", "question": "How is dengue spread?"},
        {"id": "d", "language": "sw", "question": "Malaria ni nini?"},
    ]
    path = tmp_path / "round.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    merged = merge([path], [{"language": "en", "question": "How is dengue spread"}], 10)
    assert sorted(r["id"] for r in merged) == ["health2-en-0", "health2-sw-0"]
    assert [r["question"] for r in merged if r["language"] == "en"] == ["What causes malaria?"]
