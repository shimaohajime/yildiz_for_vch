import json

import numpy as np
import pandas as pd

import src.run_polaris_analysis as polaris_analysis
from src.process_polaris_data import process_polaris_data
from src.run_polaris_analysis import build_parser, compute_nga_metrics, prepare_polaris_for_npsde


def test_process_polaris_data_writes_expected_schema(tmp_path):
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    pd.DataFrame(
        {
            "NGA": ["Latium", "Latium", "Paris Basin", ""],
            "Year": [0, 100, 0, 0],
            "Scale_1": [1.0, 2.0, 3.0, 9.0],
            "Comp": [4.0, 5.0, 6.0, 9.0],
        }
    ).to_csv(raw_path, index=False)

    result = process_polaris_data(raw_path, output_path)

    assert output_path.exists()
    assert list(result.columns) == ["NGA", "Year", "Time", "Scale", "Computation"]
    assert result["NGA"].nunique() == 2
    assert result.loc[result["NGA"] == "Latium", "Time"].tolist() == [1.0, 2.0]


def test_prepare_polaris_for_npsde_schema(tmp_path):
    input_path = tmp_path / "processed.csv"
    pd.DataFrame(
        {
            "NGA": ["Latium", "Latium"],
            "Year": [0, 100],
            "Time": [1.0, 2.0],
            "Scale": [1.0, 2.0],
            "Computation": [4.0, 5.0],
        }
    ).to_csv(input_path, index=False)

    result = prepare_polaris_for_npsde(input_path)

    assert list(result.columns) == ["Label", "Time", "x1", "x2"]
    assert result.to_dict("records") == [
        {"Label": "Latium", "Time": 1, "x1": 1.0, "x2": 4.0},
        {"Label": "Latium", "Time": 2, "x1": 2.0, "x2": 5.0},
    ]


def test_cli_defaults_match_public_interface():
    args = build_parser().parse_args([])

    assert args.input == "data/scv_processed_for_npsde.csv"
    assert args.output_dir == "scv_analysis_outputs"
    assert args.seed == 20260125


def test_compute_nga_metrics_returns_relative_paths(tmp_path, monkeypatch):
    processed = pd.DataFrame(
        {
            "Label": ["Latium", "Latium"],
            "Time": [1, 2],
            "x1": [0.0, 1.0],
            "x2": [0.0, 1.0],
        }
    )
    original = pd.DataFrame(
        {
            "NGA": ["Latium", "Latium"],
            "Year": [0, 100],
            "Time": [1, 2],
            "Scale": [0.0, 1.0],
            "Computation": [0.0, 1.0],
        }
    )

    def fake_transition_log_ratio(*args, **kwargs):
        return np.array([0.5]), np.array([-1.0]), np.array([-1.5])

    monkeypatch.setattr(
        "src.run_polaris_analysis.transition_log_ratio",
        fake_transition_log_ratio,
    )
    monkeypatch.setattr(polaris_analysis, "PROJECT_ROOT", tmp_path)

    output_dir = tmp_path / "scv_analysis_outputs"
    output_dir.mkdir()
    result = compute_nga_metrics(
        "Latium",
        npsde=object(),
        processed_df=processed,
        original_df=original,
        output_dir=output_dir,
        Nw=2,
    )

    assert result["metrics_path"].endswith("Latium_metrics.csv")
    assert result["plot_path"].endswith("Latium_aligned_plots.png")
    assert not result["metrics_path"].startswith("/")
    assert not result["plot_path"].startswith("/")
    metrics = pd.read_csv(output_dir / "Latium_metrics.csv")
    assert metrics.loc[1, "log_ratio"] == 0.5


def test_summary_json_can_use_relative_result_paths(tmp_path):
    summary = {
        "input_path": "data/scv_processed_for_npsde.csv",
        "output_dir": "scv_analysis_outputs",
        "results": [{"metrics_path": "scv_analysis_outputs/Latium_metrics.csv"}],
    }
    path = tmp_path / "analysis_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert not loaded["results"][0]["metrics_path"].startswith("/")
