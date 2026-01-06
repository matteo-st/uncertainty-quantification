#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


def _pretty_dataset(name: str) -> str:
    if name.lower() == "cifar10":
        return "CIFAR-10"
    if name.lower() == "cifar100":
        return "CIFAR-100"
    return name.replace("_", " ").title()


def _pretty_model(model: str, preprocessor: str | None) -> str:
    model_lower = model.lower()
    if model_lower == "resnet34":
        base = "ResNet-34"
    elif model_lower == "densenet121":
        base = "DenseNet-121"
    else:
        base = model.replace("_", " ").title()
    if preprocessor:
        return f"{base} ({preprocessor.upper()})"
    return base


def _pretty_postprocessor(name: str) -> str:
    return name.replace("_", " ").title()


def _fmt_fpr(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{value * 100:.2f}%"


def _fmt_auc(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{value:.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _load_summaries(results_root: Path) -> pd.DataFrame:
    summaries = list(results_root.glob("**/summary.csv"))
    frames = []
    for summary in summaries:
        df = pd.read_csv(summary)
        if df.empty:
            continue
        rel = summary.relative_to(results_root)
        if "dataset" not in df.columns and len(rel.parts) >= 3:
            df["dataset"] = rel.parts[0]
        if "postprocessor" not in df.columns and len(rel.parts) >= 3:
            df["postprocessor"] = rel.parts[2]
        if "model" not in df.columns and len(rel.parts) >= 2:
            model_preproc = rel.parts[1]
            if "_" in model_preproc:
                model_name, preproc = model_preproc.split("_", 1)
                df["model"] = model_name
                df["preprocessor"] = preproc
            else:
                df["model"] = model_preproc
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_report(results_root: Path) -> str:
    df = _load_summaries(results_root)
    if df.empty:
        raise RuntimeError(f"No summary.csv files found under {results_root}")

    key_cols = ["dataset", "model", "preprocessor", "postprocessor"]
    df = df.dropna(subset=["dataset", "model", "postprocessor"])
    df_latest = df.groupby(key_cols, dropna=False, sort=False).tail(1)

    postprocessors = sorted(df_latest["postprocessor"].dropna().unique())

    sections = [
        "# Results Summary",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## Per postprocessor",
    ]

    for post in postprocessors:
        df_post = df_latest[df_latest["postprocessor"] == post].copy()
        df_post = df_post.sort_values(["dataset", "model", "preprocessor"], kind="mergesort")

        rows = []
        for _, row in df_post.iterrows():
            rows.append(
                [
                    _pretty_dataset(str(row["dataset"])),
                    _pretty_model(str(row["model"]), row.get("preprocessor")),
                    _fmt_fpr(row.get("fpr_test")),
                    _fmt_auc(row.get("roc_auc_test")),
                ]
            )

        sections.extend(
            [
                "",
                f"### {_pretty_postprocessor(post)}",
                _markdown_table(
                    ["Dataset", "Model", "FPR (test)", "AUROC (test)"],
                    rows,
                ),
            ]
        )

    sections.append("")
    sections.append("## Across postprocessors")

    header = ["Dataset", "Model"] + [_pretty_postprocessor(p) for p in postprocessors]
    rows = []
    for (dataset, model, preprocessor), group in df_latest.groupby(
        ["dataset", "model", "preprocessor"], dropna=False, sort=False
    ):
        row = [
            _pretty_dataset(str(dataset)),
            _pretty_model(str(model), preprocessor),
        ]
        for post in postprocessors:
            match = group[group["postprocessor"] == post]
            if match.empty:
                row.append("–")
            else:
                val = match.iloc[0]
                row.append(f"{_fmt_fpr(val.get('fpr_test'))} / {_fmt_auc(val.get('roc_auc_test'))}")
        rows.append(row)

    sections.append(_markdown_table(header, rows))
    sections.append("")
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Markdown report from summary.csv files.")
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing results/",
    )
    parser.add_argument(
        "--output",
        default="results/REPORT.md",
        help="Output markdown file path",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    report = build_report(results_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
