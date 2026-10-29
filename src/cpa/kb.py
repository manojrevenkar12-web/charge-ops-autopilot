"""Retrieval over vendor manuals and EDRI runbooks (kb/*.md): BM25 over passages (one bullet or
paragraph each), filtered to the charger's own manual plus the runbook. No embedding model is needed,
which keeps the system usable without a model download; exact codes like "0x1F01" are what matter
most, and BM25 matches them exactly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import bm25s

from .settings import ROOT

MANUALS = {
    "HPC-350": "alpenstrom-hpc-350.md",
    "DC-150": "nordvolt-dc-150.md",
    "DC-60": "kestrel-dc-60.md",
    "AC-22": "volta-ac-22.md",
}
RUNBOOK = "edri-noc-runbook.md"


@dataclass(frozen=True)
class Passage:
    doc: str
    section: str
    text: str


@cache
def passages() -> tuple[Passage, ...]:
    out = []
    for f in sorted((ROOT / "kb").glob("*.md")):
        section = ""
        for block in re.split(r"\n(?=- |## |# )", f.read_text()):
            b = block.strip()
            if b.startswith("#"):
                section = b.lstrip("# ").splitlines()[0]
                rest = "\n".join(b.splitlines()[1:]).strip()
                if not rest:
                    continue
                b = rest
            out.append(Passage(f.name, section, b))
    return tuple(out)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_\-]+", text.lower())


@cache
def _index() -> tuple[bm25s.BM25, tuple[Passage, ...]]:
    ps = passages()
    idx = bm25s.BM25()
    idx.index([_tokens(f"{p.section} {p.text}") for p in ps], show_progress=False)
    return idx, ps


def search(query: str, model: str | None = None, k: int = 4) -> list[Passage]:
    idx, ps = _index()
    allowed = {RUNBOOK} | ({MANUALS[model]} if model in MANUALS else set(MANUALS.values()))
    q = _tokens(query)
    if not q:
        return []
    docs, scores = idx.retrieve([q], k=len(ps), show_progress=False)
    out = []
    for i, s in zip(docs[0], scores[0], strict=True):
        p = ps[int(i)]
        if p.doc in allowed and s > 0:
            out.append(p)
        if len(out) >= k:
            break
    return out
