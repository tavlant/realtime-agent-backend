"""Ingest every .md / .txt file under a folder into the knowledge base.

Usage:
    python scripts/ingest_docs.py docs/ --url http://localhost:8000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise SystemExit(f"no .md or .txt files under {args.folder}")
    with httpx.Client(base_url=args.url, timeout=60) as http:
        for path in files:
            response = http.post(
                "/documents",
                json={"source": path.name, "content": path.read_text(encoding="utf-8")},
            )
            response.raise_for_status()
            print(f"{path.name}: {response.json()['chunks']} chunks")


if __name__ == "__main__":
    main()
