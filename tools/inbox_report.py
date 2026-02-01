#!/usr/bin/env python3
"""
inbox_report.py
Scan PDFs in a folder and produce a CSV report:
- file size
- page count
- text-layer presence (scan suspicion)
- keyword-based category suggestion

Usage:
  python tools/inbox_report.py <folder> [--out reports/inbox_report.csv] [--sample-pages 3]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Prefer PyMuPDF (fast); fallback to pdfplumber if needed.
try:
    import fitz  # PyMuPDF
    HAVE_FITZ = True
except Exception:
    HAVE_FITZ = False

try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except Exception:
    HAVE_PDFPLUMBER = False


@dataclass
class FileRow:
    filename: str
    path: str
    ext: str
    size_mb: float
    pages: int
    text_chars_sampled: int
    text_coverage: str
    scan_suspected: bool
    score_ger: int
    score_rubric: int
    score_method: int
    score_unit: int
    category: str
    hints: str


KEYWORDS: Dict[str, List[str]] = {
    # Category A: GER / descriptors / levels
    "ger": [
        r"\bger\b", r"gemeinsame[rn]?\s+europäische[rn]?\s+referenzrahmen",
        r"referenzniveaus?", r"deskriptor", r"globalskala", r"\bA1\b", r"\bA2\b", r"\bB1\b",
        r"\bB2\b", r"\bC1\b", r"\bC2\b", r"kann\s+.*", r"skala", r"skal(en)?",
    ],
    # Category: rubrics / assessment
    "rubric": [
        r"beurteil", r"bewertung", r"kriterien", r"raster", r"rubric", r"prüfung", r"test",
        r"kompetenzraster", r"lernzielkontrolle", r"selbsteinschätzung", r"feedbackbogen"
    ],
    # Category: methods / didactics
    "method": [
        r"unterrichtsplanung", r"aviva", r"phasen", r"method", r"scaffolding",
        r"differenz", r"aufgabentyp", r"rollenspiel", r"info[-\s]?gap", r"handlungsorient",
        r"sprachhandlung", r"aktivierung", r"sicherung", r"transfer"
    ],
    # Category: units / handouts
    "unit": [
        r"handout", r"unterrichtseinheit", r"lektion", r"verlaufsplan", r"arbeitsblatt",
        r"material", r"auftrag", r"übungen", r"dialog", r"lückentext", r"spiel"
    ],
}

RE_MULTI_SPACE = re.compile(r"\s+")
RE_NONWORD = re.compile(r"[^\wäöüÄÖÜß]+")


def human_mb(n_bytes: int) -> float:
    return round(n_bytes / (1024 * 1024), 2)


def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = RE_MULTI_SPACE.sub(" ", s)
    return s.strip().lower()


def keyword_score(text: str, patterns: List[str]) -> int:
    score = 0
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            score += 1
    return score


def suggest_category(sg: int, sr: int, sm: int, su: int) -> str:
    scores = {
        "ger_norms": sg,
        "rubrics": sr,
        "methods_templates": sm,
        "exemplar_units": su,
    }
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return "unknown"
    # If ties, pick a sensible order
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], ["ger_norms","rubrics","methods_templates","exemplar_units","unknown"].index(kv[0]) if kv[0]!="unknown" else 999))
    return ordered[0][0]


def extract_sample_text_fitz(pdf_path: Path, sample_pages: int) -> Tuple[int, int, str]:
    """
    Returns: (pages_total, chars_sampled, sample_text)
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    pages_to_read = min(sample_pages, total_pages)
    chunks: List[str] = []
    chars = 0
    for i in range(pages_to_read):
        page = doc.load_page(i)
        t = page.get_text("text") or ""
        t = normalize_text(t)
        if t:
            chunks.append(t)
            chars += len(t)
    doc.close()
    return total_pages, chars, " ".join(chunks)


def extract_sample_text_pdfplumber(pdf_path: Path, sample_pages: int) -> Tuple[int, int, str]:
    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        pages_to_read = min(sample_pages, total_pages)
        chunks: List[str] = []
        chars = 0
        for i in range(pages_to_read):
            t = pdf.pages[i].extract_text() or ""
            t = normalize_text(t)
            if t:
                chunks.append(t)
                chars += len(t)
    return total_pages, chars, " ".join(chunks)


def text_coverage_label(chars: int) -> str:
    # very rough; sampled chars across first N pages
    if chars == 0:
        return "none"
    if chars < 200:
        return "very_low"
    if chars < 1000:
        return "low"
    if chars < 3000:
        return "medium"
    return "high"


def scan_folder(folder: Path, out_csv: Path, sample_pages: int) -> List[FileRow]:
    files = sorted([p for p in folder.rglob("*") if p.is_file()])
    rows: List[FileRow] = []

    for p in files:
        ext = p.suffix.lower().lstrip(".")
        size_mb = human_mb(p.stat().st_size)

        if ext != "pdf":
            # Still include docx etc. lightly, but no deep analysis
            rows.append(FileRow(
                filename=p.name,
                path=str(p),
                ext=ext,
                size_mb=size_mb,
                pages=0,
                text_chars_sampled=0,
                text_coverage="n/a",
                scan_suspected=False,
                score_ger=0,
                score_rubric=0,
                score_method=0,
                score_unit=0,
                category="non_pdf",
                hints="not a pdf"
            ))
            continue

        pages_total = 0
        chars = 0
        sample_text = ""

        try:
            if HAVE_FITZ:
                pages_total, chars, sample_text = extract_sample_text_fitz(p, sample_pages)
            elif HAVE_PDFPLUMBER:
                pages_total, chars, sample_text = extract_sample_text_pdfplumber(p, sample_pages)
            else:
                raise RuntimeError("Neither PyMuPDF (fitz) nor pdfplumber is installed.")
        except Exception as e:
            rows.append(FileRow(
                filename=p.name,
                path=str(p),
                ext=ext,
                size_mb=size_mb,
                pages=0,
                text_chars_sampled=0,
                text_coverage="error",
                scan_suspected=False,
                score_ger=0,
                score_rubric=0,
                score_method=0,
                score_unit=0,
                category="error",
                hints=f"read_error: {type(e).__name__}"
            ))
            continue

        cov = text_coverage_label(chars)
        scan_suspected = (chars == 0)

        sg = keyword_score(sample_text, KEYWORDS["ger"])
        sr = keyword_score(sample_text, KEYWORDS["rubric"])
        sm = keyword_score(sample_text, KEYWORDS["method"])
        su = keyword_score(sample_text, KEYWORDS["unit"])

        cat = suggest_category(sg, sr, sm, su)

        hints = []
        if scan_suspected:
            hints.append("scan_suspected(textlayer=0)")
        if size_mb >= 30:
            hints.append("large_pdf")
        if pages_total >= 100:
            hints.append("many_pages")
        if cat == "unknown":
            hints.append("no_keywords_hit")

        rows.append(FileRow(
            filename=p.name,
            path=str(p),
            ext=ext,
            size_mb=size_mb,
            pages=pages_total,
            text_chars_sampled=chars,
            text_coverage=cov,
            scan_suspected=scan_suspected,
            score_ger=sg,
            score_rubric=sr,
            score_method=sm,
            score_unit=su,
            category=cat,
            hints=";".join(hints) if hints else ""
        ))

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "ext", "size_mb", "pages",
            "text_chars_sampled", "text_coverage", "scan_suspected",
            "score_ger", "score_rubric", "score_method", "score_unit",
            "category", "hints", "path"
        ])
        for r in rows:
            w.writerow([
                r.filename, r.ext, r.size_mb, r.pages,
                r.text_chars_sampled, r.text_coverage, str(r.scan_suspected),
                r.score_ger, r.score_rubric, r.score_method, r.score_unit,
                r.category, r.hints, r.path
            ])

    return rows


def print_summary(rows: List[FileRow], out_csv: Path) -> None:
    pdfs = [r for r in rows if r.ext == "pdf" and r.category not in ("error",)]
    scans = [r for r in pdfs if r.scan_suspected]
    by_cat: Dict[str, int] = {}
    for r in pdfs:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1

    def topn(cat: str, n: int = 8) -> List[FileRow]:
        xs = [r for r in pdfs if r.category == cat]
        xs.sort(key=lambda r: (-(r.score_ger + r.score_rubric + r.score_method + r.score_unit), -r.pages, -r.size_mb))
        return xs[:n]

    print("\n=== Inbox Report Summary ===")
    print(f"CSV: {out_csv}")
    print(f"Total files: {len(rows)} | PDFs: {len(pdfs)} | Suspected scans (textlayer=0): {len(scans)}")
    print("By category:", ", ".join([f"{k}={v}" for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])]))

    if scans:
        scans_sorted = sorted(scans, key=lambda r: (-r.size_mb, -r.pages))
        print("\n--- Suspected scan PDFs (check OCR) ---")
        for r in scans_sorted[:10]:
            print(f"  {r.size_mb:>6} MB | {r.pages:>4} p | {r.filename}")

    for cat in ("ger_norms", "rubrics", "methods_templates", "exemplar_units"):
        xs = topn(cat, 6)
        if xs:
            print(f"\n--- Top candidates: {cat} ---")
            for r in xs:
                score = r.score_ger if cat == "ger_norms" else r.score_rubric if cat == "rubrics" else r.score_method if cat == "methods_templates" else r.score_unit
                print(f"  score={score:>2} | {r.size_mb:>6} MB | {r.pages:>4} p | {r.filename}")

    print("\nNext move suggestion:")
    print("  - Move top 6–10 PDFs from ger_norms/rubrics/methods into docs/20_kernel/... and ingest ONLY those first.")
    print("  - Keep scan_suspected PDFs aside; handle later with OCR if needed.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="Folder with raw PDFs (recursive)")
    ap.add_argument("--out", default="", help="Output CSV path (default: reports/inbox_report_<timestamp>.csv)")
    ap.add_argument("--sample-pages", type=int, default=3, help="How many first pages to sample for text/keywords")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.out).expanduser().resolve() if args.out else (Path("reports") / f"inbox_report_{ts}.csv").resolve()

    if not HAVE_FITZ and not HAVE_PDFPLUMBER:
        print("ERROR: Need PyMuPDF (fitz) or pdfplumber installed.", file=sys.stderr)
        print("Try: pip install pymupdf", file=sys.stderr)
        return 3

    rows = scan_folder(folder, out_csv, args.sample_pages)
    print_summary(rows, out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
