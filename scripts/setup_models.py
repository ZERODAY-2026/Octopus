"""One-time SETUP script: the only place that is allowed to use the network.

Downloads the embedding model into the local Hugging Face cache and checks that
the Ollama model is present. Run it before the app; the app itself runs offline.
"""

from __future__ import annotations

import os
import subprocess
import sys

for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(var, None)  # setup is allowed to download

from sentence_transformers import SentenceTransformer  # noqa: E402

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3n:e4b")


def main() -> int:
    print(f"[setup] downloading embedding model {EMBEDDING_MODEL} ...")
    SentenceTransformer(EMBEDDING_MODEL)
    print(f"[setup] pulling Ollama model {OLLAMA_MODEL} ...")
    rc = subprocess.call(["ollama", "pull", OLLAMA_MODEL])
    if rc != 0:
        print("[setup] `ollama pull` failed - is Ollama installed and running?", file=sys.stderr)
        return rc
    subprocess.call(["ollama", "list"])
    print("[setup] done. From now on the app needs no network access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
