"""Prepare the public frozen assets and passage index for the local demo."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
HF = "https://huggingface.co"
DATA = f"{HF}/datasets/sricharan1234/tiny-aya-hallucination-guard-data/resolve/17990d9e2456b468e840cf7aa102ac93e7dbb167"
MODEL = f"{HF}/sricharan1234/tiny-aya-hallucination-guard/resolve/ee5846e589583ca77e3c79b0b4a1a9c29d588f59"
ASSETS = {
    "guard_fire.npz": (f"{MODEL}/guard_fire.npz",
                       "1d919bdabad5fce3d87087e3aa882765e3532739ab08eae7241a515dca196a5e"),
    "health_chunks.jsonl": (f"{DATA}/data/health_chunks.jsonl",
                            "829c37655505a67f74cfd3a24ac03329551d6bae1c12174c195c4a7c913e3b71"),
}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_asset(target, override=None):
    url, expected = ASSETS[target.name]
    if override is None and target.is_file() and sha256(target) == expected:
        return
    if override is not None and override.resolve() == target.resolve():
        return
    with NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        if override is None:
            with urlopen(url, timeout=90) as response, temp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            if sha256(temp) != expected:
                raise ValueError(f"Downloaded artifact has an unexpected SHA-256: {target.name}")
        else:
            shutil.copyfile(override, temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def build_vectors(out, device):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from build_rag import ENCODER, embed

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with (out / "health_chunks.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to(device).eval()
    vectors = embed([row["text"] for row in rows], tokenizer, encoder, "passage: ", batch_size=32)
    with NamedTemporaryFile(dir=out, delete=False) as handle:
        temp = Path(handle.name)
    try:
        with temp.open("wb") as handle:
            np.save(handle, vectors.float().cpu().numpy())
        temp.replace(out / "health_vectors.npy")
    finally:
        temp.unlink(missing_ok=True)
    print(f"Indexed {len(rows)} passages on {device}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "space_build")
    parser.add_argument("--guard", type=Path, help="Use a local guard instead of the frozen public artifact.")
    parser.add_argument("--chunks", type=Path, help="Use a local health corpus instead of the frozen public artifact.")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--fetch-only", action="store_true", help="Stage the corpus and guard without building vectors.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prepare_asset(args.out / "guard_fire.npz", args.guard)
    prepare_asset(args.out / "health_chunks.jsonl", args.chunks)
    if not args.fetch_only:
        build_vectors(args.out, args.device)
    print(args.out.resolve())


if __name__ == "__main__":
    main()
