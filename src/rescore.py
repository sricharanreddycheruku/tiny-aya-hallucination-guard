"""Recompute correctness fields of a saved answers file with the current scorer."""
import json
import sys

from scoring import exact_match, is_wrong, normalize, token_f1

path = sys.argv[1]
records = [json.loads(line) for line in open(path, encoding="utf-8")]
with open(path, "w", encoding="utf-8") as out:
    for r in records:
        language, answer, golds = r["language"], r["answer"], r["golds"]
        r["f1"] = token_f1(answer, golds, language)
        r["exact"] = exact_match(answer, golds, language)
        r["contains"] = any(normalize(g, language) in normalize(answer, language) for g in golds)
        r["wrong"] = is_wrong(answer, golds, language)
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
