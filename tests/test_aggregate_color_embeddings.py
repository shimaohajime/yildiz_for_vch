import pandas as pd
import pytest

from src.aggregate_color_embeddings import (
    aggregate_by_artist_year,
    compute_principal_components,
    detect_embedding_columns,
    majority_vote,
)


def test_detect_embedding_columns_sorts_numeric_suffixes():
    columns = [
        "other_column",
        "color_embedding_10",
        "color_embedding_2",
        "color_embedding_1",
        "color_embedding",
    ]
    detected = detect_embedding_columns(columns, prefix="color_embedding")
    assert detected == [
        "color_embedding",
        "color_embedding_1",
        "color_embedding_2",
        "color_embedding_10",
    ]


def test_compute_principal_components_adds_columns():
    df = pd.DataFrame(
        {
            "color_embedding_1": [0.0, 1.0, 2.0],
            "color_embedding_2": [1.0, 2.0, 3.0],
            "artist": ["A", "A", "B"],
            "date": [1900, 1900, 1950],
            "genre": ["Landscape", "Landscape", "Portrait"],
            "style": ["Impressionism", "Impressionism", "Realism"],
        }
    )
    result = compute_principal_components(df, ["color_embedding_1", "color_embedding_2"])
    assert {"PC1", "PC2"} <= set(result.columns)
    assert result["PC1"].mean() == pytest.approx(0.0, abs=1e-9)
    assert result["PC2"].mean() == pytest.approx(0.0, abs=1e-9)


def test_majority_vote_prefers_most_frequent():
    series = pd.Series(["A", "B", "A", "C", "A"])
    assert majority_vote(series) == "A"


def test_aggregate_by_artist_year_means_numeric_and_modes_categorical():
    df = pd.DataFrame(
        {
            "artist": ["A", "A", "B"],
            "date": [1900, 1900, 1900],
            "genre": ["Landscape", "Landscape", "Portrait"],
            "style": ["Impressionism", "Impressionism", "Realism"],
            "PC1": [1.0, 3.0, 2.0],
            "PC2": [2.0, 4.0, 5.0],
        }
    )
    aggregated = aggregate_by_artist_year(df)
    assert list(aggregated.columns) == ["artist", "date", "genre", "style", "PC1", "PC2"]
    artist_a = aggregated.loc[aggregated["artist"] == "A"].iloc[0]
    assert artist_a["PC1"] == pytest.approx(2.0)
    assert artist_a["genre"] == "Landscape"


def test_full_pipeline_drops_title_and_filename_columns():
    df = pd.DataFrame(
        {
            "artist": ["A", "A"],
            "date": [1900, 1900],
            "genre": ["Landscape", "Landscape"],
            "style": ["Impressionism", "Impressionism"],
            "title": ["Work 1", "Work 2"],
            "filename": ["w1.jpg", "w2.jpg"],
            "color_embedding_1": [0.1, 0.2],
            "color_embedding_2": [0.3, 0.4],
            "color_embedding_3": [0.5, 0.6],
        }
    )
    components = compute_principal_components(
        df, ["color_embedding_1", "color_embedding_2", "color_embedding_3"]
    )
    components = components.drop(columns=["title", "filename"])
    aggregated = aggregate_by_artist_year(components)
    assert "title" not in aggregated.columns
    assert "filename" not in aggregated.columns
