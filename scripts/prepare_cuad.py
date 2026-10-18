"""Turn a locally downloaded CUAD into a Contracter evaluation set.

CUAD is NOT committed to this repository (licence: CC BY 4.0, Hendrycks et al.,
NeurIPS 2021). Point this script at your own copy:

    uv run python scripts/prepare_cuad.py --cuad-dir ../datasets/CUAD_v1 \
        --contracts 40 --seed 7 --out eval/

Outputs (the contract text itself is never copied):
    eval/cuad_subset.jsonl   ground truth in Contracter's schema
    eval/cuad_docs.json      {slug: path-to-contract-text}  (used by --docs-map)
    eval/cuad_manifest.json  contract titles + seed, so the subset is reproducible

Then:
    uv run python scripts/run_evaluation.py --dataset eval/cuad_subset.jsonl \
        --docs-map eval/cuad_docs.json --embedder minilm --judge
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

#: CUAD category -> (question kept verbatim from CUAD, canonical clause types)
CATEGORIES: dict[str, list[str]] = {
    "Governing Law": ["governing_law", "dispute_resolution"],
    "Expiration Date": ["termination", "deadline"],
    "Effective Date": ["deadline"],
    "Agreement Date": ["deadline"],
    "Renewal Term": ["renewal", "termination"],
    "Notice Period To Terminate Renewal": ["notice", "termination"],
    "Termination For Convenience": ["termination"],
    "Anti-Assignment": ["assignment"],
    "Cap On Liability": ["liability"],
    "Non-Compete": ["non_compete"],
    "Exclusivity": ["obligation"],
    "Audit Rights": ["obligation"],
    "Insurance": ["obligation"],
    "Liquidated Damages": ["penalty"],
    "Covenant Not To Sue": ["obligation"],
}


#: CUAD's own prompts ("Highlight the parts (if any) of this contract related to ...")
#: are annotation instructions, not questions a user would ask. Use natural wording.
NATURAL_QUESTIONS: dict[str, str] = {
    "Governing Law": "Which law governs this agreement?",
    "Expiration Date": "When does this agreement expire or end?",
    "Effective Date": "What is the effective date of this agreement?",
    "Agreement Date": "On what date is this agreement dated?",
    "Renewal Term": "Does the agreement renew, and for how long?",
    "Notice Period To Terminate Renewal": "What notice is required to stop the agreement from renewing?",
    "Termination For Convenience": "Can a party terminate the agreement without cause?",
    "Anti-Assignment": "Can a party assign the agreement without the other party's consent?",
    "Cap On Liability": "Is there a cap on liability?",
    "Non-Compete": "Is there a non-compete restriction?",
    "Exclusivity": "Does the agreement grant exclusivity?",
    "Audit Rights": "Does either party have audit rights?",
    "Insurance": "Is insurance required?",
    "Liquidated Damages": "Are there liquidated damages or penalties?",
    "Covenant Not To Sue": "Is there a covenant not to sue?",
}


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:50]


def find_contract_text(cuad_dir: Path, title: str) -> Path | None:
    """Locate the plain-text (preferred) or PDF version of a contract by title."""
    for folder, suffix in (("full_contract_txt", ".txt"), ("full_contract_pdf", ".pdf")):
        for candidate in cuad_dir.rglob(f"{title}{suffix}"):
            return candidate
        for candidate in (cuad_dir / folder).rglob(f"*{suffix}") if (cuad_dir / folder).exists() else []:
            if candidate.stem.lower() == title.lower():
                return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cuad-dir", type=Path, required=True)
    parser.add_argument("--contracts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("eval"))
    parser.add_argument("--include", nargs="*", default=[], help="Extra contract titles to force in (e.g. your Energy XXI test file)")
    args = parser.parse_args()

    json_path = next(args.cuad_dir.rglob("CUAD_v1.json"), None)
    if json_path is None:
        raise SystemExit("CUAD_v1.json not found under --cuad-dir")
    data = json.loads(json_path.read_text(encoding="utf-8"))["data"]

    usable = [(entry["title"], find_contract_text(args.cuad_dir, entry["title"])) for entry in data]
    usable = [(t, p) for t, p in usable if p is not None]
    by_title = {entry["title"]: entry for entry in data}

    rng = random.Random(args.seed)
    forced = [t for t, _ in usable if any(x.lower() in t.lower() for x in args.include)]
    pool = [t for t, _ in usable if t not in forced]
    rng.shuffle(pool)
    chosen = forced + pool[: max(0, args.contracts - len(forced))]
    paths = dict(usable)

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    docs_map: dict[str, str] = {}
    for title in chosen:
        slug = slugify(title)
        docs_map[slug] = str(paths[title].resolve())
        for para in by_title[title]["paragraphs"]:
            for qa in para["qas"]:
                category = qa["id"].split("__")[-1]
                if category not in CATEGORIES:
                    continue
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                answerable = bool(answers) and not qa.get("is_impossible", False)
                rows.append(
                    {
                        "id": f"{slug}:{slugify(category)}",
                        "question": NATURAL_QUESTIONS.get(category, qa["question"]),
                        "cuad_question": qa["question"],
                        "reference_answer": " ".join(answers)[:1500] if answerable else "",
                        "answerable": answerable,
                        "document": slug,
                        "relevant_clause_types": CATEGORIES[category],
                        "relevant_keywords": [a[:120] for a in answers[:2]],
                        "gold_passages": answers[:3],
                        "category": "cuad_" + slugify(category),
                    }
                )

    (args.out / "cuad_subset.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (args.out / "cuad_docs.json").write_text(json.dumps(docs_map, indent=2), encoding="utf-8")
    (args.out / "cuad_manifest.json").write_text(
        json.dumps({"seed": args.seed, "contracts": chosen}, indent=2), encoding="utf-8"
    )
    unanswerable = sum(1 for r in rows if not r["answerable"])
    print(f"{len(chosen)} contracts, {len(rows)} questions ({unanswerable} unanswerable) -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
