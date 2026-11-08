"""Collect the demo's data files: corpus, passage vectors and the frozen guard."""
import json
import shutil
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

from build_rag import ENCODER, embed


def main():
    out = Path("../space_build")
    out.mkdir(exist_ok=True)
    shutil.copy("../data/health_chunks.jsonl", out)
    shutil.copy("../runs/guard/guard_fire.npz", out)
    rows = [json.loads(line) for line in open("../data/health_chunks.jsonl", encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    encoder = AutoModel.from_pretrained(ENCODER).to("cuda").eval()
    vectors = embed([row["text"] for row in rows], tokenizer, encoder, "passage: ")
    np.save(out / "health_vectors.npy", vectors.float().cpu().numpy())
    print(vectors.shape)


if __name__ == "__main__":
    main()
