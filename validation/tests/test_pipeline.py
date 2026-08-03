from __future__ import annotations

from pathlib import Path

import msprime
import pandas as pd
import pytest
import yaml

import infer
from validation.configuration import (
    ConfigurationError,
    load_config,
    preflight_config,
)
from validation.run import ResultExistsError, run_validation
from validation.scripts.point_accuracy_common import ValidationResult
from validation.scripts.point_accuracy_common import clip_dataframe_to_region


def _dump_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _base_config(
    truth: Path,
    tsinfer: Path,
    singer_dir: Path,
    *,
    singer_prefix: str = "singer_",
) -> dict:
    return {
        "truth": {
            "trees": str(truth),
            "ne": 10_000,
            "haplotypes": 4,
        },
        "methods": {
            "gfn": {"sample_prefix": "arg_"},
            "tsinfer": {"inferred_trees": str(tsinfer)},
            "singer": {
                "input_dir": str(singer_dir),
                "sample_prefix": singer_prefix,
            },
        },
        "validation": {
            "skip": 1,
            "max_pairs": 2,
            "pair_seed": 7,
            "xlim": [0, 16],
            "ylim": "0,8",
            "xlim_log": [-4, 1.5],
            "ylim_log": "-4,1.5",
        },
    }


def _touch_inputs(tmp_path: Path, output_root: Path, experiment: str):
    truth = tmp_path / "truth.trees"
    truth.write_text("truth", encoding="utf-8")
    tsinfer = tmp_path / "tsinfer.trees"
    tsinfer.write_text("tsinfer", encoding="utf-8")
    singer_dir = tmp_path / "singer"
    singer_dir.mkdir()
    (singer_dir / "singer_000001.trees").write_text(
        "singer", encoding="utf-8"
    )
    gfn_dir = output_root / experiment / "gfn"
    gfn_dir.mkdir(parents=True)
    (gfn_dir / "arg_000001.trees").write_text("gfn", encoding="utf-8")
    return truth, tsinfer, singer_dir


def test_load_config_resolves_paths_and_default_gfn(tmp_path):
    experiment = "trial-01"
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, experiment
    )
    config_path = _dump_yaml(
        tmp_path / "validation.yaml",
        _base_config(truth, tsinfer, singer_dir),
    )

    config = load_config(
        config_path, experiment, output_root=output_root
    )

    assert config.truth.trees == truth.resolve()
    assert config.methods["gfn"].mode == "samples"
    assert config.methods["gfn"].input_dir == (
        output_root / experiment / "gfn"
    ).resolve()
    assert config.methods["tsinfer"].mode == "trees"
    assert config.methods["singer"].mode == "samples"
    assert config.validation.xlim == "0,16"
    preflight_config(config)


def test_config_parses_region_and_rejects_invalid_bounds(tmp_path):
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, "trial-region"
    )
    raw = _base_config(truth, tsinfer, singer_dir)
    raw["validation"]["region"] = [25, 75]
    config = load_config(
        _dump_yaml(tmp_path / "region.yaml", raw),
        "trial-region",
        output_root=output_root,
    )
    assert config.validation.region == (25.0, 75.0)

    raw["validation"]["region"] = [75, 25]
    with pytest.raises(ConfigurationError, match="0 <= left < right"):
        load_config(
            _dump_yaml(tmp_path / "bad-region.yaml", raw),
            "trial-region",
            output_root=output_root,
        )


def test_region_filter_clips_boundary_segments_and_weights():
    frame = pd.DataFrame(
        {
            "chr": ["1", "1", "1"],
            "start": [0, 40, 80],
            "end": [40, 80, 120],
            "Simulated": [1.0, 2.0, 3.0],
            "PosteriorMean": [1.0, 2.0, 3.0],
            "PosteriorMedian": [1.0, 2.0, 3.0],
            "len": [40.0, 40.0, 40.0],
        }
    )
    clipped = clip_dataframe_to_region(frame, 25.0, 95.0)
    assert clipped["start"].tolist() == [25.0, 40.0, 80.0]
    assert clipped["end"].tolist() == [40.0, 80.0, 95.0]
    assert clipped["len"].sum() == pytest.approx(70.0)


def test_config_rejects_unknown_and_ambiguous_method_keys(tmp_path):
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, "trial"
    )
    raw = _base_config(truth, tsinfer, singer_dir)
    raw["methods"]["gfn"]["inferred_trees"] = str(tsinfer)
    config_path = _dump_yaml(tmp_path / "ambiguous.yaml", raw)
    with pytest.raises(ConfigurationError, match="exactly one input mode"):
        load_config(config_path, "trial", output_root=output_root)

    raw = _base_config(truth, tsinfer, singer_dir)
    raw["validation"]["typo"] = 1
    config_path = _dump_yaml(tmp_path / "unknown.yaml", raw)
    with pytest.raises(ConfigurationError, match="unknown key"):
        load_config(config_path, "trial", output_root=output_root)

    raw = _base_config(truth, tsinfer, singer_dir)
    del raw["methods"]["singer"]
    config = load_config(
        _dump_yaml(tmp_path / "two_methods.yaml", raw),
        "trial",
        output_root=output_root,
    )
    assert list(config.methods) == ["gfn", "tsinfer"]
    preflight_config(config)

    raw = _base_config(truth, tsinfer, singer_dir)
    raw["methods"] = {}
    config_path = _dump_yaml(tmp_path / "missing.yaml", raw)
    with pytest.raises(ConfigurationError, match="at least one method"):
        load_config(config_path, "trial", output_root=output_root)


@pytest.mark.parametrize("name", ["../escape", "has/slash", "", "."])
def test_config_rejects_unsafe_experiment_names(tmp_path, name):
    config_path = _dump_yaml(tmp_path / "empty.yaml", {})
    with pytest.raises(ValueError, match="experiment name"):
        load_config(config_path, name, output_root=tmp_path / "outputs")


def test_preflight_reports_missing_method_output(tmp_path):
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, "trial"
    )
    (singer_dir / "singer_000001.trees").unlink()
    config = load_config(
        _dump_yaml(
            tmp_path / "validation.yaml",
            _base_config(truth, tsinfer, singer_dir),
        ),
        "trial",
        output_root=output_root,
    )
    with pytest.raises(ConfigurationError, match="found no singer_"):
        preflight_config(config)


def _fake_runner(name: str, seen: list[tuple[str, object, object]]):
    def run(args, *, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = output_dir / "metrics.tsv"
        artifact.write_text("metric\tvalue\n", encoding="utf-8")
        seen.append((name, args.truth_trees, args.max_pairs))
        return ValidationResult(
            method=name,
            method_label=name.upper(),
            metrics={
                "n_segments": 2,
                "total_length": 100.0,
                "weighted_mse": 0.25,
                "weighted_rmse": 0.5,
                "weighted_mae": 0.4,
                "weighted_bias": 0.1,
                "legacy_mseall": 0.3,
            },
            legacy_mse=0.3,
            artifacts=(artifact,),
        )

    return run


def test_runner_dispatches_all_methods_and_protects_results(tmp_path):
    experiment = "trial"
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, experiment
    )
    config = load_config(
        _dump_yaml(
            tmp_path / "validation.yaml",
            _base_config(truth, tsinfer, singer_dir),
        ),
        experiment,
        output_root=output_root,
    )
    seen: list[tuple[str, object, object]] = []
    runners = {
        name: _fake_runner(name, seen)
        for name in ("gfn", "tsinfer", "singer")
    }

    results_dir = run_validation(config, method_runners=runners)

    assert [entry[0] for entry in seen] == ["gfn", "tsinfer", "singer"]
    assert all(entry[1] == truth.resolve() for entry in seen)
    assert all(entry[2] == 2 for entry in seen)
    summary = pd.read_csv(results_dir / "summary.tsv", sep="\t")
    assert summary["method"].tolist() == ["gfn", "tsinfer", "singer"]
    assert (results_dir / "weighted_rmse.png").is_file()
    assert (results_dir / "run_manifest.yaml").is_file()

    with pytest.raises(ResultExistsError):
        run_validation(config, method_runners=runners)

    replaced = run_validation(config, force=True, method_runners=runners)
    assert replaced == results_dir
    assert not list((output_root / experiment).glob(".results-*"))


def test_runner_dispatches_configured_method_subset(tmp_path):
    experiment = "trial-subset"
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, experiment
    )
    raw = _base_config(truth, tsinfer, singer_dir)
    del raw["methods"]["singer"]
    config = load_config(
        _dump_yaml(tmp_path / "validation.yaml", raw),
        experiment,
        output_root=output_root,
    )
    seen: list[tuple[str, object, object]] = []
    runners = {
        name: _fake_runner(name, seen)
        for name in ("gfn", "tsinfer", "singer")
    }

    results_dir = run_validation(config, method_runners=runners)

    assert [entry[0] for entry in seen] == ["gfn", "tsinfer"]
    summary = pd.read_csv(results_dir / "summary.tsv", sep="\t")
    assert summary["method"].tolist() == ["gfn", "tsinfer"]
    manifest = yaml.safe_load(
        (results_dir / "run_manifest.yaml").read_text(encoding="utf-8")
    )
    assert list(manifest["methods"]) == ["gfn", "tsinfer"]
    assert set(manifest["results"]) == {"gfn", "tsinfer"}


def test_failed_method_does_not_publish_results(tmp_path):
    experiment = "failure"
    output_root = tmp_path / "outputs"
    truth, tsinfer, singer_dir = _touch_inputs(
        tmp_path, output_root, experiment
    )
    config = load_config(
        _dump_yaml(
            tmp_path / "validation.yaml",
            _base_config(truth, tsinfer, singer_dir),
        ),
        experiment,
        output_root=output_root,
    )
    seen: list[tuple[str, object, object]] = []
    runners = {
        "gfn": _fake_runner("gfn", seen),
        "tsinfer": _fake_runner("tsinfer", seen),
        "singer": lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("SINGER failed")
        ),
    }

    with pytest.raises(RuntimeError, match="SINGER failed"):
        run_validation(config, method_runners=runners)

    experiment_root = output_root / experiment
    assert not (experiment_root / "results").exists()
    assert not list(experiment_root.glob(".results-*"))


def test_tiny_tree_sequence_integration(tmp_path):
    experiment = "tiny"
    output_root = tmp_path / "outputs"
    tree_sequence = msprime.sim_ancestry(
        samples=4,
        ploidy=1,
        sequence_length=100,
        recombination_rate=0,
        population_size=10_000,
        random_seed=23,
    )
    truth = tmp_path / "truth.trees"
    tree_sequence.dump(truth)
    tsinfer = tmp_path / "tsinfer.trees"
    tree_sequence.dump(tsinfer)

    gfn_dir = output_root / experiment / "gfn"
    gfn_dir.mkdir(parents=True)
    tree_sequence.dump(gfn_dir / "arg_000001.trees")
    singer_dir = tmp_path / "singer"
    singer_dir.mkdir()
    tree_sequence.dump(singer_dir / "singer_000001.trees")

    config = load_config(
        _dump_yaml(
            tmp_path / "validation.yaml",
            _base_config(truth, tsinfer, singer_dir),
        ),
        experiment,
        output_root=output_root,
    )
    results_dir = run_validation(config)

    summary = pd.read_csv(results_dir / "summary.tsv", sep="\t")
    assert summary["method"].tolist() == ["gfn", "tsinfer", "singer"]
    assert summary["weighted_rmse"].abs().max() == pytest.approx(0)
    for method in ("gfn", "tsinfer", "singer"):
        assert (results_dir / method / "metrics.tsv").is_file()
        assert (results_dir / method / "time_summary.tsv").is_file()
    manifest = yaml.safe_load(
        (results_dir / "run_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert set(manifest["results"]) == {"gfn", "tsinfer", "singer"}


def test_inference_experiment_path_and_cli_exclusion():
    output_dir = Path(infer.resolve_inference_output_dir(experiment="trial"))
    assert output_dir.parts[-4:] == (
        "validation",
        "output",
        "trial",
        "gfn",
    )
    assert infer.resolve_inference_output_dir() == "inferred_args"
    with pytest.raises(ValueError, match="either output_dir or experiment"):
        infer.resolve_inference_output_dir(
            output_dir="custom", experiment="trial"
        )
    with pytest.raises(SystemExit):
        infer.build_parser().parse_args(
            [
                "--checkpoint",
                "best.pt",
                "--output-dir",
                "custom",
                "--experiment",
                "trial",
            ]
        )


def test_inference_checkpoint_directory_resolves_best(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert infer.resolve_checkpoint_path(run_dir) == str(checkpoint.resolve())
    assert infer.resolve_checkpoint_path(checkpoint.parent) == str(
        checkpoint.resolve()
    )
    assert infer.resolve_checkpoint_path(checkpoint) == str(checkpoint.resolve())

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No best.pt"):
        infer.resolve_checkpoint_path(empty_dir)


def test_inference_manifest_records_resolved_context(tmp_path):
    class State:
        log_reward = 1.0
        accumulated_log_prior = 0.5

    class TreeSequence:
        num_trees = 2
        num_edges = 3

    class Environment:
        def save_to_tree_sequence(self, state, output_path):
            return TreeSequence()

        def get_arg_sequence_segments(self, state):
            return {
                "breakpoints": [0, 10],
                "num_segments": 1,
                "recombination_events": [],
            }

    manifest = infer.build_manifest(
        checkpoint=tmp_path / "best.pt",
        metadata={"epoch": 4, "best_loss": 0.25},
        seed=7,
        random_spec=None,
        output_dir=tmp_path / "experiment" / "gfn",
        env=Environment(),
        rollout_outputs={
            "states": [State()],
            "log_paths_pf": infer.torch.tensor([[0.1]]),
            "log_paths_pb": infer.torch.tensor([[0.2]]),
        },
        trajectories=[[object()]],
        dataset_path=tmp_path / "input.vcf",
        experiment="experiment",
    )

    assert manifest["experiment"] == "experiment"
    assert manifest["dataset_path"] == str((tmp_path / "input.vcf").resolve())
    assert manifest["output_dir"] == str(
        (tmp_path / "experiment" / "gfn").resolve()
    )
    assert manifest["output_count"] == 1
