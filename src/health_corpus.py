"""Public-health passages from Wikipedia (CC BY-SA 4.0) in four languages."""
import argparse
import json
import time
from pathlib import Path

import requests

TOPICS = [
    "Malaria", "Tuberculosis", "Diabetes", "Dengue fever", "Cholera", "Typhoid fever", "Hepatitis B",
    "HIV/AIDS", "Measles", "Poliomyelitis", "Rabies", "Tetanus", "Pneumonia", "Asthma", "Hypertension",
    "Anemia", "Diarrhea", "Dehydration", "Snakebite", "Influenza", "COVID-19", "Smallpox", "Leprosy",
    "Chickenpox", "Breastfeeding", "Vaccination", "Malnutrition", "Heat stroke", "Stroke", "Cancer",
    "Epilepsy", "Lymphatic filariasis", "Chikungunya", "Yellow fever", "Ebola", "Oral rehydration therapy",
    "Vitamin A deficiency", "Pregnancy", "Hand washing", "Obesity",
]
LANGUAGES = ("en", "te", "bn", "sw")
MIN_CHARS, MAX_CHARS = 300, 900
HEADERS = {"User-Agent": "lexsi-research/0.1"}


def api(language, **params):
    url = f"https://{language}.wikipedia.org/w/api.php"
    for _ in range(6):
        reply = requests.get(url, params={"action": "query", "format": "json", "redirects": 1, **params}, headers=HEADERS, timeout=30)
        if reply.status_code != 429:
            break
        time.sleep(int(reply.headers.get("Retry-After", 30)))
    reply.raise_for_status()
    time.sleep(1.5)
    return next(iter(reply.json()["query"]["pages"].values()))


def chunks(text):
    parts, current = [], ""
    for block in text.split("\n"):
        block = block.strip()
        if not block or block.startswith("="):
            continue
        if current and len(current) + len(block) > MAX_CHARS:
            parts.append(current)
            current = ""
        current = f"{current} {block}".strip()
    if current:
        parts.append(current)
    return [p[:MAX_CHARS] for p in parts if len(p) >= MIN_CHARS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/health_chunks.jsonl")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(LANGUAGES, 0)
    articles = dict.fromkeys(LANGUAGES, 0)
    with open(args.out, "w", encoding="utf-8") as out:
        for topic in TOPICS:
            links = api("en", titles=topic, prop="langlinks", lllimit=500).get("langlinks", [])
            titles = {link["lang"]: link["*"] for link in links}
            titles["en"] = topic
            for language in LANGUAGES:
                if language not in titles:
                    continue
                page = api(language, titles=titles[language], prop="extracts|info", explaintext=1, inprop="url")
                pieces = chunks(page.get("extract", ""))
                articles[language] += bool(pieces)
                for index, text in enumerate(pieces):
                    out.write(json.dumps({
                        "language": language, "topic": topic, "title": page["title"], "url": page.get("fullurl"),
                        "chunk": index, "text": text,
                    }, ensure_ascii=False) + "\n")
                    counts[language] += 1
    print("articles", articles, "chunks", counts)


if __name__ == "__main__":
    main()
