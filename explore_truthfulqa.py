#!/usr/bin/env python3

import argparse
from collections import Counter

import pandas as pd


DEFAULT_PATH = (
    "hf://datasets/LM-Polygraph/truthfulqa/continuation/"
    "test-00000-of-00001.parquet"
)


def extract_question(prompt: str) -> str:
    marker = "\n\nQ: "
    idx = prompt.rfind(marker)
    start = idx + len(marker) if idx != -1 else prompt.rfind("Q: ") + 3
    end = prompt.rfind("\nA:")
    return prompt[start:end].strip() if end != -1 else prompt[start:].strip()


def extract_prefix(prompt: str, question: str) -> str:
    needle = f"Q: {question}\nA:"
    return prompt[: prompt.rfind(needle)]


def summarize(df: pd.DataFrame, samples: int) -> None:
    questions = df["input"].map(extract_question)
    prefixes = [extract_prefix(prompt, question) for prompt, question in zip(df["input"], questions)]
    choice_counts = df["output"].map(len)
    flat_choices = [str(choice) for choices in df["output"] for choice in choices]

    print("Shape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}: {df[col].dtype}")

    print("\nNull counts:")
    print(df.isna().sum().to_string())

    print("\nQuestion length stats:")
    print(questions.str.len().describe().round(2).to_string())

    print("\nCandidate answer count stats:")
    print(choice_counts.describe().round(2).to_string())

    print("\nCandidate answer count distribution:")
    print(choice_counts.value_counts().sort_index().to_string())

    print("\nPrompt prefix variants:")
    prefix_counts = Counter(prefixes)
    for idx, (prefix, count) in enumerate(prefix_counts.items(), start=1):
        preview = prefix.replace("\n", " ")[:120]
        print(f"- variant {idx}: {count} rows | len={len(prefix)} | preview={preview!r}")

    flat_choice_lengths = pd.Series([len(choice) for choice in flat_choices])
    print("\nFlattened answer length stats:")
    print(flat_choice_lengths.describe().round(2).to_string())

    print(f"\nSample rows ({min(samples, len(df))}):")
    for i in range(min(samples, len(df))):
        row = df.iloc[i]
        choices = [str(choice) for choice in row["output"]]
        print(f"\nRow {i}")
        print("Question:", questions.iloc[i])
        print("Choices:", choices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the TruthfulQA continuation parquet file.")
    parser.add_argument("--path", default=DEFAULT_PATH, help="Parquet path or local file path.")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample rows to print.")
    args = parser.parse_args()

    print("Loading:", args.path)
    df = pd.read_parquet(args.path)
    summarize(df, samples=args.samples)


if __name__ == "__main__":
    main()
