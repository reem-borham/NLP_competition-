#!/usr/bin/env python3
"""
Weak aspect labeling for Arabic reviews (ABSA bootstrapping).

Creates:
  - aspect_* boolean columns (multi-label)
  - primary_aspect (single-label, highest score)
  - aspect_score_* integer columns

This is a fast baseline to generate pseudo-labels you can later refine or
replace with a supervised aspect classifier.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL_RE = re.compile(r"\u0640")
NON_LETTER_RE = re.compile(r"[^\w\u0600-\u06FF]+", flags=re.UNICODE)


def normalize_arabic(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = TATWEEL_RE.sub("", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
        .replace("ة", "ه")
    )
    text = NON_LETTER_RE.sub(" ", text)
    return " ".join(text.split()).strip().lower()


def _compile_keywords(keywords: Iterable[str]) -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    for kw in keywords:
        kw_norm = normalize_arabic(kw)
        if not kw_norm:
            continue
        # Word-ish boundary: handle Arabic + latin/underscore tokens from \w
        patterns.append(re.compile(rf"(?<!\w){re.escape(kw_norm)}(?!\w)"))
    return patterns


@dataclass(frozen=True)
class AspectConfig:
    name: str
    keywords: Tuple[str, ...]


DEFAULT_ASPECTS: Tuple[AspectConfig, ...] = (
    AspectConfig(
        name="service",
        keywords=(
            "خدمه",
            "الخدمه",
            "الموظفين",
            "موظف",
            "الاستقبال",
            "الريسبشن",
            "تسجيل الدخول",
            "تسجيل خروج",
            "تشيك ان",
            "تشيك اوت",
            "تعامل",
            "بشوش",
            "وقح",
            "تعاون",
            "مساعده",
            "استجابه",
            "سريعه",
            "بطيئه",
        ),
    ),
    AspectConfig(
        name="logistics",
        keywords=(
            "حجز",
            "الحجز",
            "حجوزات",
            "تطبيق",
            "ابلكيشن",
            "الدفع",
            "فلوس",
            "سعر",
            "اسعار",
            "فاتوره",
            "الغاء",
            "تعديل",
            "تأكيد",
            "تاكيد",
            "الدخول",
            "الخروج",
            "انتظار",
            "طابور",
            "مواصلات",
            "نقل",
            "المطار",
            "باركينج",
            "موقف",
            "موقف سيارات",
        ),
    ),
    AspectConfig(
        name="location",
        keywords=(
            "موقع",
            "الموقع",
            "المنطقه",
            "قريب",
            "بعيد",
            "جنب",
            "وسط",
            "اطلاله",
            "اطلالة",
            "منظر",
            "بحر",
            "شاطئ",
            "سيتي سنتر",
            "وسط البلد",
            "مواصلات",
            "مكان",
            "حى",
            "حي",
        ),
    ),
    AspectConfig(
        name="cleaning",
        keywords=(
            "نظافه",
            "النظافه",
            "نظيف",
            "وسخ",
            "متسخ",
            "تعقيم",
            "ريحه",
            "رائحه",
            "رائحة",
            "حمام",
            "الحمام",
            "مناشف",
            "ملاءات",
            "غبار",
            "حشرات",
            "صراصير",
        ),
    ),
)


def score_aspects(
    text: str,
    aspect_patterns: Dict[str, List[re.Pattern]],
    *,
    return_matches: bool = False,
) -> Tuple[Dict[str, int], Optional[Dict[str, List[str]]]]:
    text_norm = normalize_arabic(text)
    scores: Dict[str, int] = {}
    matches: Dict[str, List[str]] = {}
    for aspect, patterns in aspect_patterns.items():
        aspect_score = 0
        aspect_matches: List[str] = []
        for pattern in patterns:
            if pattern.search(text_norm):
                aspect_score += 1
                if return_matches:
                    aspect_matches.append(pattern.pattern)
        scores[aspect] = aspect_score
        if return_matches:
            matches[aspect] = aspect_matches
    return scores, (matches if return_matches else None)


def add_aspect_labels(
    df: pd.DataFrame,
    *,
    text_col: str,
    aspects: Tuple[AspectConfig, ...] = DEFAULT_ASPECTS,
    unknown_label: str = "unknown",
) -> pd.DataFrame:
    aspect_patterns = {a.name: _compile_keywords(a.keywords) for a in aspects}

    def _primary(scores: Dict[str, int]) -> str:
        best_aspect = unknown_label
        best_score = 0
        for aspect, score in scores.items():
            if score > best_score:
                best_aspect = aspect
                best_score = score
        return best_aspect if best_score > 0 else unknown_label

    scores_series = df[text_col].fillna("").map(lambda t: score_aspects(t, aspect_patterns)[0])
    for aspect in aspect_patterns.keys():
        df[f"aspect_score_{aspect}"] = scores_series.map(lambda s, a=aspect: int(s.get(a, 0)))
        df[f"aspect_{aspect}"] = df[f"aspect_score_{aspect}"] > 0
    df["primary_aspect"] = scores_series.map(_primary)
    return df


def guess_text_column(df: pd.DataFrame) -> str:
    candidates = [
        "review",
        "text",
        "content",
        "comment",
        "sentence",
        "Review",
        "Text",
        "Content",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    # fallback: first object-like column with high average length
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("Could not find a text column; pass --text-col explicitly.")
    best_col = max(object_cols, key=lambda c: df[c].fillna("").astype(str).map(len).mean())
    return best_col


def load_from_kagglehub(dataset: str, file_path: str) -> pd.DataFrame:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "kagglehub is not installed. Install with: pip install kagglehub[pandas-datasets]"
        ) from exc

    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        file_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Input CSV path.")
    src.add_argument("--kaggle-dataset", help="Kaggle dataset slug, e.g. abedkhooli/arabic-100k-reviews.")
    parser.add_argument("--kaggle-file", default="", help="File path inside the Kaggle dataset (required with --kaggle-dataset).")
    parser.add_argument("--text-col", default="", help="Text column name (auto-detect if omitted).")
    parser.add_argument("--out", default="labeled_reviews.csv", help="Output CSV path.")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        if not args.kaggle_file:
            raise SystemExit("--kaggle-file is required with --kaggle-dataset")
        df = load_from_kagglehub(args.kaggle_dataset, args.kaggle_file)

    text_col = args.text_col or guess_text_column(df)
    df = add_aspect_labels(df, text_col=text_col)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out} (text_col={text_col})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

