import json
from pathlib import Path

import pytest
import tskit
import yaml

try:
    from arg.sample_cwr_refinement import (
        build_parser,
        run_cwr_refinement_sampler,
    )
except ImportError:
    from sample_cwr_refinement import (
        build_parser,
        run_cwr_refinement_sampler,
    )


def _write_vcf(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.vcf"
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "##contig=<ID=1,length=100>",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0",
                "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0|1",
                "1\t40\t.\tG\tT\t.\tPASS\t.\tGT\t1|0",
                "1\t80\t.\tC\tA\t.\tPASS\t.\tGT\t0|1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_source_tree_sequence(tmp_path: Path) -> Path:
    tables = tskit.TableCollection(sequence_length=100)
    tables.time_units = "generations"
    sample_flags = tskit.NODE_IS_SAMPLE
    tables.nodes.add_row(flags=sample_flags, time=0.0)
    tables.nodes.add_row(flags=sample_flags, time=0.0)
    parent = tables.nodes.add_row(flags=0, time=10.0)
    tables.edges.add_row(left=0.0, right=100.0, parent=parent, child=0)
    tables.edges.add_row(left=0.0, right=100.0, parent=parent, child=1)
    site = tables.sites.add_row(position=9.0, ancestral_state="A")
    tables.mutations.add_row(site=site, node=1, derived_state="C")
    site = tables.sites.add_row(position=39.0, ancestral_state="G")
    tables.mutations.add_row(site=site, node=0, derived_state="T")
    site = tables.sites.add_row(position=79.0, ancestral_state="C")
    tables.mutations.add_row(site=site, node=1, derived_state="A")
    tables.sort()
    path = tmp_path / "source.trees"
    tables.tree_sequence().dump(path)
    return path


def _write_config(tmp_path: Path) -> Path:
    dataset_path = _write_vcf(tmp_path)
    source_arg_path = _write_source_tree_sequence(tmp_path)
    config = {
        "dataset_path": str(dataset_path),
        "output_path": str(tmp_path / "configured-output"),
        "refinement": {
            "arg_path": str(source_arg_path),
            "requests": [
                {
                    "id": "target",
                    "genomic_range": [0, 100],
                    "cut_event_index": 0,
                }
            ],
        },
        "training": {"seed": 19},
        "environment": {
            "bp_per_blocks": 1,
            "effective_population_size": 10_000,
            "mutation_rate": 2e-8,
            "recombination_rate": 0.0,
        },
        "reward": {"constant": 0.0},
    }
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_sampler_exports_requested_tree_count_without_model(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "samples"

    manifest = run_cwr_refinement_sampler(
        config_path,
        num_trees=2,
        output_dir=output_dir,
    )

    assert manifest["mode"] == "cwr_prior_local_refinement"
    assert manifest["uses_model"] is False
    assert manifest["num_trees_per_request"] == 2
    assert manifest["output_count"] == 2
    request = manifest["requests"][0]
    assert request["id"] == "target"
    assert len(request["outputs"]) == 2
    for index in (1, 2):
        output_path = output_dir / "target" / f"arg_{index:06d}.trees"
        tree_sequence = tskit.load(output_path)
        assert tree_sequence.sequence_length == 100
        assert tree_sequence.num_samples == 2
        assert tree_sequence.num_edges > 0

    saved_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["output_count"] == 2


def test_sampler_seed_is_reproducible(tmp_path):
    config_path = _write_config(tmp_path)
    first = run_cwr_refinement_sampler(
        config_path,
        num_trees=2,
        output_dir=tmp_path / "first",
        seed=31,
    )
    second = run_cwr_refinement_sampler(
        config_path,
        num_trees=2,
        output_dir=tmp_path / "second",
        seed=31,
    )

    first_outputs = first["requests"][0]["outputs"]
    second_outputs = second["requests"][0]["outputs"]
    assert [
        row["topology_digest"] for row in first_outputs
    ] == [
        row["topology_digest"] for row in second_outputs
    ]
    assert [
        row["local_cwr_log_prior"] for row in first_outputs
    ] == pytest.approx(
        [row["local_cwr_log_prior"] for row in second_outputs]
    )


def test_sampler_refuses_to_overwrite_by_default(tmp_path):
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "samples"
    run_cwr_refinement_sampler(
        config_path,
        num_trees=1,
        output_dir=output_dir,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_cwr_refinement_sampler(
            config_path,
            num_trees=1,
            output_dir=output_dir,
        )


def test_cli_requires_num_trees():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "config.yaml"])
