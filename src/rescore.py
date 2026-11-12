"""Compare saved answer labels with the current scorer without running a model."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from scoring import exact_match, is_wrong, normalize, token_f1


def rescore(records):
    groups = defaultdict(lambda: {"n": 0, "original_wrong": 0, "revised_wrong": 0, "changed": 0})
    revised = []
    for row in records:
        language, answer, golds = row["language"], row["answer"], row["golds"]
        wrong = is_wrong(answer, golds, language)
        for key in ("all", language):
            counts = groups[key]
            counts["n"] += 1
            counts["original_wrong"] += row["wrong"]
            counts["revised_wrong"] += wrong
            counts["changed"] += row["wrong"] != wrong
        revised.append(dict(
            row, wrong=wrong, f1=token_f1(answer, golds, language), exact=exact_match(answer, golds, language),
            contains=any(normalize(g, language) in normalize(answer, language) for g in golds),
        ))
    return revised, dict(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", type=Path, help="Write revised labels to a new file.")
    target.add_argument("--summary-only", action="store_true", help="Keep the saved answer log untouched.")
    args = parser.parse_args()
    with args.answers.open(encoding="utf-8") as handle:
        revised, report = rescore(json.loads(line) for line in handle)
    print(json.dumps(report, indent=2))
    if args.summary_only:
        return
    out = args.out or args.answers
    out.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=out.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for row in revised:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
