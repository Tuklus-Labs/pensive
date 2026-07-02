"""Generate the 200 synthetic sample atoms the backfill test runs against.

Deterministic (seeded) so the committed ``synthetic_atoms.jsonl`` is exactly
reproducible. These are SYNTHETIC by construction -- invented project slugs,
invented timestamps, invented text -- so unlike the real corpus export they are
safe to commit as a fixture (the real-corpus dev store is never tracked; see the
task-11 privacy rule).

Each record is one ``srcExport`` element in the shape ``backfill`` consumes::

    {
      "sourceId":  <original id or path>  -> provenance.sourceRef,
      "text":      <atom body>            -> atom.text (required),
      "kind":      "atom" | "narrative"   -> atom.kind,
      "project":   <slug> | null          -> atom.project,
      "occurredAt": <unix seconds> | null -> atom.occurred_at,
      "importance": <float>               -> atom.importance,
      "tags":      ["src:<name>", ...]     -> facets (key='tag', value verbatim)
    }

The text is stitched from fragments that deliberately contain entities the v2
``MegaExtractor(REAL_DATA_PATTERNS)`` sees -- project slugs, dates, tech terms --
so the backfill test can assert entity facets land in the pinned convention
(key='entity', value=lowercase surface form). Regenerate with::

    python3 daemon/test/ingest/fixtures/gen_synthetic_atoms.py
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "synthetic_atoms.jsonl"
N = 200
SEED = 0x9E3779B9  # fixed golden-ratio seed; any stable value works

# Project slugs the NL_PATTERNS literal set recognizes as ('<slug>', 'project').
# Text carrying one of these is guaranteed a deterministic entity facet.
KNOWN_PROJECTS = ["pensive", "aegis", "kairos", "lenora", "hermes", "kesagake"]

# Tech terms NL_PATTERNS recognizes as ('<term>', 'tech').
KNOWN_TECH = ["rocm", "pytorch", "faiss", "redis", "docker", "sqlalchemy"]

# Non-entity project slugs (for atoms deliberately WITHOUT a known entity, so the
# fixture also exercises the ~83%-no-entity real-query reality).
PLAIN_PROJECTS = ["frostguard", "suppressor-sim", "case-pressure", "git-dig"]

# Filler sentence stems with no extractable entity, to pad realistic-length bodies.
FILLER = [
    "the reasoning stalled until the constraint was restated plainly",
    "an earlier assumption about the buffer sizing turned out inverted",
    "the fix held after the third rebuild and a clean cache",
    "we agreed the simpler path was worth the small regression",
    "the note is kept for the next session to pick up cold",
    "throughput settled once the contention on the shared lane eased",
    "the decision was recorded so it would not be relitigated",
    "a quiet failure mode surfaced only under sustained load",
]


def _body(rng, project_entity, tech_entity, with_date, repeat_entity):
    """Stitch a realistic multi-clause atom body from seeded fragments."""
    parts = []
    if project_entity:
        parts.append(
            f"working notes on {project_entity} came out of the afternoon pass"
        )
        if repeat_entity:
            # Same entity twice -> still exactly one entity facet (idempotency).
            parts.append(f"the {project_entity} thread stayed the focus throughout")
    if tech_entity:
        parts.append(f"the {tech_entity} path was the one that finally held")
    if with_date:
        # YYYY-MM-DD -> ('<date>', 'date') under BASE_PATTERNS.
        y = rng.randint(2023, 2026)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        parts.append(f"logged against {y:04d}-{m:02d}-{d:02d} for the record")
    # Always add 1-3 filler clauses so bodies read like real notes.
    for _ in range(rng.randint(1, 3)):
        parts.append(rng.choice(FILLER))
    rng.shuffle(parts)
    return ". ".join(parts) + "."


def generate():
    rng = random.Random(SEED)
    base_time = 1_700_000_000  # ~2023-11, a plausible historical anchor
    records = []
    for i in range(N):
        # ~40% carry a known project entity, ~30% a tech entity, ~25% a date.
        has_proj_entity = rng.random() < 0.40
        has_tech_entity = rng.random() < 0.30
        has_date = rng.random() < 0.25
        repeat_entity = has_proj_entity and (rng.random() < 0.30)

        project_entity = rng.choice(KNOWN_PROJECTS) if has_proj_entity else None
        tech_entity = rng.choice(KNOWN_TECH) if has_tech_entity else None

        # Project facet: a known slug when the body carries that entity, else a
        # plain (non-entity) slug, and occasionally null (project-less atom).
        if project_entity is not None:
            project = project_entity
        elif rng.random() < 0.85:
            project = rng.choice(PLAIN_PROJECTS)
        else:
            project = None

        kind = "narrative" if rng.random() < 0.15 else "atom"

        # occurredAt spread over ~3 years so the time prior has real spread; a
        # few atoms leave it null to exercise the created_at fallback.
        if rng.random() < 0.90:
            occurred_at = base_time + rng.randint(0, 3 * 365 * 86_400)
        else:
            occurred_at = None

        importance = round(rng.choice([0.0, 0.0, 0.0, 0.2, 0.5, 0.8]), 3)

        # 1-2 src: tags per atom, drawn from a modest pool so some tags cluster
        # (>=2 atoms share a tag) -- the atom-corpus sibling-retrieval proxy in
        # the gate relies on such clusters existing.
        n_tags = rng.randint(1, 2)
        tags = [f"src:topic-{rng.randint(0, 39):02d}" for _ in range(n_tags)]
        tags = sorted(set(tags))

        text = _body(rng, project_entity, tech_entity, has_date, repeat_entity)

        records.append({
            "sourceId": f"engram-{i:05d}",
            "text": text,
            "kind": kind,
            "project": project,
            "occurredAt": occurred_at,
            "importance": importance,
            "tags": tags,
        })
    return records


def main():
    records = generate()
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} synthetic atoms to {OUT}")


if __name__ == "__main__":
    main()
