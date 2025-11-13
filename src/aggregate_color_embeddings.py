"""Utilities for aggregating WikiArt color embeddings at the artist-year level."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.decomposition import PCA


def detect_embedding_columns(columns: Iterable[str], prefix: str) -> list[str]:
    """Return color embedding columns sorted by their numeric suffix."""
    candidates: list[tuple[int, str]] = []
    for name in columns:
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            try:
                order = int(suffix.strip("_")) if suffix else 0
            except ValueError:
                order = float("inf")
            candidates.append((order, name))
    # Sort by detected numeric suffix while keeping lexicographic fallback.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in candidates]


def compute_principal_components(
    df: pd.DataFrame, embedding_columns: Sequence[str], n_components: int = 2
) -> pd.DataFrame:
    """Attach the first ``n_components`` principal components to ``df``."""
    if len(embedding_columns) < n_components:
        raise ValueError(
            "Not enough embedding columns were found to compute the requested number of principal components."
        )

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.loc[:, embedding_columns])
    result = df.copy()
    for index in range(n_components):
        result[f"PC{index + 1}"] = components[:, index]
    return result


def majority_vote(series: pd.Series) -> object:
    """Return the most frequent value in ``series`` (ties resolved by ``Series.mode`` order)."""
    mode = series.mode(dropna=True)
    if mode.empty:
        return pd.NA
    return mode.iloc[0]


def aggregate_by_artist_year(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate principal components and categorical metadata by artist and year."""
    required_columns = {"artist", "date", "PC1", "PC2", "genre", "style"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {sorted(missing)}")

    aggregated = (
        df.groupby(["artist", "date"], dropna=False)
        .agg({"PC1": "mean", "PC2": "mean", "genre": majority_vote, "style": majority_vote})
        .reset_index()
    )

    ordered_columns = ["artist", "date", "genre", "style", "PC1", "PC2"]
    return aggregated.loc[:, ordered_columns]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PCA over color embedding variables from the WikiArtVectors dataset and "
            "aggregate the first two components by artist-year."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the WikiArt color representation CSV file.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination path for the aggregated artist-year PCA CSV file.",
    )
    parser.add_argument(
        "--embedding-prefix",
        default="color_embedding",
        help="Prefix used to automatically detect color embedding columns.",
    )
    parser.add_argument(
        "--embedding-columns",
        nargs="+",
        help=(
            "Explicit list of embedding column names. Overrides automatic detection when provided."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_path: Path = args.input_path
    output_path: Path = args.output_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file '{input_path}' was not found. Please download the 'Color Representations' "
            "dataset from https://github.com/bhargavvader/WikiArtVectors and provide its path."
        )

    df = pd.read_csv(input_path)

    if args.embedding_columns is not None:
        embedding_columns = list(args.embedding_columns)
    else:
        embedding_columns = detect_embedding_columns(df.columns, args.embedding_prefix)

    if not embedding_columns:
        raise ValueError(
            "Unable to identify color embedding columns. Provide them explicitly with --embedding-columns."
        )

    df_with_pcs = compute_principal_components(df, embedding_columns, n_components=2)

    # Drop columns that should not appear in the aggregated output but might remain in intermediate data.
    columns_to_drop = [col for col in ("title", "filename") if col in df_with_pcs.columns]
    if columns_to_drop:
        df_with_pcs = df_with_pcs.drop(columns=columns_to_drop)

    aggregated = aggregate_by_artist_year(df_with_pcs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
