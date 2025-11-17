"""Utilities for aggregating WikiArt color embeddings at the artist-year level."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def aggregate_embeddings_by_artist_year(
    df: pd.DataFrame, embedding_columns: Sequence[str]
) -> pd.DataFrame:
    """Aggregate embeddings and categorical metadata by artist and year."""
    required_columns = {"artist", "date", "genre", "style"}
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(f"Input dataframe is missing required columns: {sorted(missing)}")

    # Check that all embedding columns exist
    missing_emb = set(embedding_columns) - set(df.columns)
    if missing_emb:
        raise KeyError(f"Missing embedding columns: {sorted(missing_emb)}")

    # Drop rows with missing embeddings
    mask = df.loc[:, embedding_columns].notna().all(axis=1)
    if not mask.any():
        raise ValueError("No rows contain complete embedding data.")
    df = df.loc[mask].copy()

    # Aggregate: mean for embeddings, majority vote for categorical
    agg_dict = {col: "mean" for col in embedding_columns}
    agg_dict.update({"genre": majority_vote, "style": majority_vote})

    aggregated = (
        df.groupby(["artist", "date"], dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    # Filter artists with >= 10 years
    counts = aggregated.groupby("artist")["date"].transform("nunique")
    aggregated = aggregated.loc[counts >= 10].reset_index(drop=True)

    return aggregated


def standardize_embeddings(
    df: pd.DataFrame, embedding_columns: Sequence[str]
) -> pd.DataFrame:
    """Standardize embedding columns to zero mean and unit variance."""
    result = df.copy()
    scaler = StandardScaler()
    result.loc[:, embedding_columns] = scaler.fit_transform(result.loc[:, embedding_columns])
    return result


def compute_principal_components(
    df: pd.DataFrame, embedding_columns: Sequence[str], n_components: int = 2
) -> pd.DataFrame:
    """Attach the first ``n_components`` principal components to ``df``.
    
    Assumes embeddings are already standardized.
    """
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

    if input_path.suffix.lower() in {".feather", ".ft"}:
        try:
            df = pd.read_feather(input_path)
        except ImportError as exc:
            raise ImportError(
                "Reading Feather files requires the 'pyarrow' dependency. "
                "Install it with `pip install pyarrow` and retry."
            ) from exc
    else:
        df = pd.read_csv(input_path)

    required_raw_columns = {"artist", "date"}
    missing_raw = required_raw_columns - set(df.columns)
    if missing_raw:
        raise KeyError(f"Input file is missing required columns: {sorted(missing_raw)}")

    df = df.copy()
    df["artist"] = df["artist"].astype(str).str.strip()
    suspicious_artists = {"", "unknown", "anonymous", "no artist", "n/a", "unnamed"}
    invalid_mask = (
        df["artist"].isna()
        | df["artist"].str.lower().isin(suspicious_artists)
        | df["artist"].str.startswith("#")
    )
    if invalid_mask.any():
        df = df.loc[~invalid_mask].copy()

    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if args.embedding_columns is not None:
        embedding_columns = list(args.embedding_columns)
    else:
        embedding_columns = detect_embedding_columns(df.columns, args.embedding_prefix)

    if not embedding_columns:
        raise ValueError(
            "Unable to identify color embedding columns. Provide them explicitly with --embedding-columns."
        )

    # Step 1: Aggregate by artist-year (this also filters to artists with >= 10 years)
    aggregated = aggregate_embeddings_by_artist_year(df, embedding_columns)

    # Step 2: Standardize the aggregated embeddings
    standardized = standardize_embeddings(aggregated, embedding_columns)

    # Step 3: Apply PCA on standardized embeddings
    result = compute_principal_components(standardized, embedding_columns, n_components=2)

    # Select final columns and drop intermediate embedding columns
    final_columns = ["artist", "date", "genre", "style", "PC1", "PC2"]
    result = result.loc[:, final_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
