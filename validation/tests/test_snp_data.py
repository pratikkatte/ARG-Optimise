"""Observation ingestion tests use temporary bundles, never modify real data."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import tskit

from env.snp_data import SNPData, load_snp_dataset


REPO = Path(__file__).resolve().parents[2]
REP0 = REPO / "validation/datasets/sim_5k_mr20/rep0"


@pytest.fixture
def bundle(tmp_path):
    metadata = {
        "schema_version": 1, "mutation_model": "infinite_sites", "mutation_positions": "continuous",
        "mutation_alphabet": "ACGT", "vcf_ref": "known simulated ancestral allele; no external reference genome",
        "parameters": {"sequence_length": 10, "contig_id": "1", "num_samples": 4},
        "num_sites": 2, "num_mutations": 2, "num_haplotypes": 4,
        "summary": {"sequence_length_bp": 10, "num_sites": 2, "num_mutations": 2,
                    "num_haplotypes": 4, "num_individuals": 2},
        "files": {"vcf": "data.vcf", "position_map": "positions.tsv", "ground_truth_trees": "absent.trees"},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "data.vcf").write_text(
        '##fileformat=VCFv4.2\n##contig=<ID=1,length=10>\n'
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tbeta\talpha\n'
        '1\t1\t10\tA\tG\t.\tPASS\t.\tDP:GT\t8:1|1\t8:0|0\n'
        '1\t2\t20\tC\tT\t.\tPASS\t.\tDP:GT\t8:0|0\t8:1|0\n'
    )
    # Map order is deliberately different from VCF order: join by site ID.
    (tmp_path / "positions.tsv").write_text(
        'site_id\tposition_zero_based\tvcf_position_one_based\n20\t0.2\t2\n10\t0.1\t1\n')
    return tmp_path


def edit_metadata(bundle, change):
    path = bundle / "metadata.json"
    metadata = json.loads(path.read_text())
    change(metadata)
    path.write_text(json.dumps(metadata))


def test_loader_preserves_phasing_coordinates_and_ids_without_trees(bundle):
    data = load_snp_dataset(bundle)
    np.testing.assert_array_equal(data.genotypes, [[1, 0], [1, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(data.positions, [0.1, 0.2])
    assert data.haplotype_ids == ("beta:0", "beta:1", "alpha:0", "alpha:1")
    assert data.site_ids == (10, 20)
    assert data.ancestral_states == ("A", "C") and data.derived_states == ("G", "T")
    assert data.sequence_length == 10 and data.num_variants == 2 and data.num_haplotypes == 4
    assert data.genotypes.dtype == np.uint8 and data.positions.dtype == np.float64
    assert not data.genotypes.flags.writeable and not data.positions.flags.writeable
    assert not list(bundle.glob("*.trees"))


def test_shifted_vcf_contig_and_positions_do_not_change_physical_span(bundle):
    vcf = bundle / "data.vcf"
    vcf.write_text(vcf.read_text().replace("length=10", "length=101").replace("1\t1\t10", "1\t100\t10").replace("1\t2\t20", "1\t101\t20"))
    (bundle / "positions.tsv").write_text('site_id\tposition_zero_based\tvcf_position_one_based\n10\t0.1\t100\n20\t0.2\t101\n')
    assert load_snp_dataset(bundle).sequence_length == 10


def test_zero_snps_retains_sample_rows_and_physical_length(bundle):
    def empty(meta):
        for obj in (meta, meta["summary"]):
            obj["num_sites"] = obj["num_mutations"] = 0
    edit_metadata(bundle, empty)
    vcf = bundle / "data.vcf"
    vcf.write_text("\n".join(line for line in vcf.read_text().splitlines() if line.startswith("#")) + "\n")
    (bundle / "positions.tsv").write_text('site_id\tposition_zero_based\tvcf_position_one_based\n')
    data = load_snp_dataset(bundle)
    assert data.genotypes.shape == (4, 0) and data.positions.shape == (0,)
    assert data.sequence_length == 10


@pytest.mark.parametrize("filename,old,new,error", [
    ("data.vcf", "8:1|1", "8:1|.", "phased diploid"),
    ("data.vcf", "8:1|1", "8:1/1", "phased diploid"),
    ("data.vcf", "8:1|1", "8:1", "phased diploid"),
    ("data.vcf", "8:1|1", "8:1|2", "phased diploid"),
    ("data.vcf", "8:1|1", "8:0|0", "segregating"),
    ("data.vcf", "\tA\tG\t", "\tA\tG,T\t", "biallelic"),
    ("data.vcf", "\tDP:GT\t", "\tDP:AD\t", "GT field"),
    ("data.vcf", "\tPASS\t", "\tLowQual\t", "filtered"),
    ("data.vcf", "1\t2\t20", "2\t2\t20", "contig"),
    ("data.vcf", "1\t2\t20", "1\t2\t10", "duplicate VCF"),
    ("data.vcf", "\tbeta\talpha", "\tbeta\tbeta", "sample names"),
    ("positions.tsv", "20\t0.2\t2", "20\t0.2\t3", "mismatch"),
    ("positions.tsv", "20\t0.2\t2", "10\t0.2\t2", "duplicate position-map"),
    ("positions.tsv", "20\t0.2\t2", "20\t0.1\t2", "strictly increasing"),
    ("positions.tsv", "20\t0.2\t2", "20\t0.05\t2", "strictly increasing"),
    ("positions.tsv", "20\t0.2\t2", "20\t10\t2", "within"),
    ("positions.tsv", "20\t0.2\t2", "20\t-1\t2", "within"),
    ("positions.tsv", "20\t0.2\t2", "20\tnan\t2", "finite"),
    ("positions.tsv", "20\t0.2\t2", "20\tinf\t2", "finite"),
    ("positions.tsv", "20\t0.2\t2\n", "", "mismatch"),
    ("data.vcf", "##contig=<ID=1,length=10>", "##contig=<ID=1,length=10>\n##contig=<ID=2,length=10>", "exactly"),
])
def test_invalid_observations_are_rejected(bundle, filename, old, new, error):
    path = bundle / filename
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, new))
    with pytest.raises(ValueError, match=error):
        load_snp_dataset(bundle)


@pytest.mark.parametrize("field,value,error", [
    ("num_haplotypes", 6, "does not match"),
    ("num_mutations", 3, "does not match"),
    ("vcf_ref", "unknown", "known ancestral REF"),
    ("mutation_model", "JC69", "infinite-sites"),
    ("schema_version", 2, "schema_version"),
])
def test_metadata_contract(bundle, field, value, error):
    edit_metadata(bundle, lambda meta: meta.update({field: value}))
    with pytest.raises(ValueError, match=error):
        load_snp_dataset(bundle)


def test_summary_length_mismatch_is_rejected(bundle):
    edit_metadata(bundle, lambda meta: meta["summary"].update(sequence_length_bp=11))
    with pytest.raises(ValueError, match="sequence_length_bp"):
        load_snp_dataset(bundle)


def test_binary_validation_precedes_uint8_cast():
    with pytest.raises(ValueError, match="binary"):
        SNPData([[256], [1]], [0.1], 1, (0,), ("A",), ("G",), ("a", "b"))


@pytest.mark.skipif(not REP0.is_dir(), reason="local rep0 dataset is not installed")
def test_rep0_exact_data_and_loading_without_ground_truth(tmp_path):
    metadata = json.loads((REP0 / "metadata.json").read_text())
    for filename in ("metadata.json", metadata["files"]["vcf"], metadata["files"]["position_map"]):
        shutil.copyfile(REP0 / filename, tmp_path / filename)
    data = load_snp_dataset(tmp_path)
    ts = tskit.load(REP0 / metadata["files"]["ground_truth_trees"])
    np.testing.assert_array_equal(data.genotypes, ts.genotype_matrix().T)
    np.testing.assert_array_equal(data.positions, ts.tables.sites.position)
    assert data.genotypes.shape == (16, 95) and data.sequence_length == 5000
    assert tuple(data.positions[71:73]) == (4266.426820184104, 4266.499721854925)
    assert np.all(np.floor(data.positions[71:73]) == 4266)


@pytest.mark.parametrize("incompatible,rate", [(False, 0.1), (True, 0.1), (False, 0)])
def test_cli_strict_json_and_read_only_inputs(bundle, incompatible, rate):
    tables = tskit.TableCollection(10)
    tables.time_units = "generations"
    for time in (0, 0, 0, 0, 1, 1, 2):
        tables.nodes.add_row(time=time, flags=tskit.NODE_IS_SAMPLE if time == 0 else 0)
    pairs = ((0, 2), (1, 3)) if incompatible else ((0, 1), (2, 3))
    for parent, children in zip((4, 5), pairs):
        for child in children:
            tables.edges.add_row(0, 10, parent, child)
        tables.edges.add_row(0, 10, 6, parent)
    tables.sort()
    candidate = bundle / "candidate.trees"
    tables.tree_sequence().dump(candidate)
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    command = [sys.executable, str(REPO / "validation/scripts/score_infinite_sites.py"),
               "--replicate-dir", str(bundle), "--trees", str(candidate), "--mutation-rate", str(rate),
               "--sample-nodes", "0", "1", "2", "3"]
    completed = subprocess.run(command, cwd=bundle, capture_output=True, text=True, check=True)
    def invalid_constant(value):
        raise AssertionError(f"nonstandard JSON constant {value}")
    report = json.loads(completed.stdout, parse_constant=invalid_constant)
    assert report["num_haplotypes"] == 4 and report["num_variants"] == 2
    assert report["zero_likelihood"] == (incompatible or rate == 0)
    assert (report["log_likelihood"] is None) == report["zero_likelihood"]
    assert report["incompatible_site_ids"] == ([10] if incompatible else [])
    assert {p.name: p.read_bytes() for p in bundle.iterdir()} == before
    invalid = subprocess.run(command + ["--mutation-rate", "-1"], cwd=bundle, capture_output=True, text=True)
    assert invalid.returncode == 2 and "mutation_rate" in invalid.stderr and not invalid.stdout
