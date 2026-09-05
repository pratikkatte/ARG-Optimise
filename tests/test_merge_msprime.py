import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import tskit

from validation.scripts import merge_msprime


def read_fasta(path):
    records = {}
    name = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:]
            records[name] = []
        else:
            records[name].append(line)
    return {name: "".join(parts) for name, parts in records.items()}


class MergeMsprimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.output_root = Path(cls.tempdir.name) / "first"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            merge_msprime.main(["--output-root", str(cls.output_root)])
        cls.stdout = stdout.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()

    def test_default_cli_reports_scaled_rates(self):
        self.assertIn("theta=40", self.stdout)
        self.assertIn("rho=4", self.stdout)

    def test_default_outputs_have_expected_dimensions(self):
        tree_path = self.output_root / "trees" / "sim_poc_easy_0.trees"
        ts = tskit.load(tree_path)
        self.assertEqual(ts.sequence_length, 10_000)
        self.assertEqual(ts.num_samples, 8)

        fasta = read_fasta(
            self.output_root / "fasta" / "sim_poc_easy_0.fa"
        )
        self.assertEqual(list(fasta), [f"hap{index:03d}" for index in range(8)])
        for sequence in fasta.values():
            self.assertEqual(len(sequence), 10_000)
            self.assertLessEqual(set(sequence), set("ACGT"))

    def test_vcf_has_four_diploid_samples_and_unique_positions(self):
        vcf_path = self.output_root / "vcf" / "sim_poc_easy_0.vcf"
        lines = vcf_path.read_text(encoding="utf-8").splitlines()
        header = next(line for line in lines if line.startswith("#CHROM"))
        self.assertEqual(header.split("\t")[9:], ["spl0", "spl1", "spl2", "spl3"])
        positions = [int(line.split("\t")[1]) for line in lines if not line.startswith("#")]
        self.assertEqual(len(positions), len(set(positions)))
        self.assertTrue(all(1 <= position <= 10_000 for position in positions))

    def test_pairwise_time_tracks_cover_the_sequence(self):
        time_dir = self.output_root / "tcoalmsp" / "rep0"
        paths = sorted(time_dir.glob("sim_poc_easy_0_spls*.tc"))
        self.assertEqual(len(paths), 28)
        for path in paths:
            rows = [line.split("\t") for line in path.read_text().splitlines()]
            self.assertEqual(float(rows[0][0]), 0)
            self.assertEqual(float(rows[-1][1]), 10_000)

    def test_seed_reproduces_tree_sequence_and_fasta(self):
        second_root = Path(self.tempdir.name) / "second"
        with contextlib.redirect_stdout(io.StringIO()):
            merge_msprime.main(["--output-root", str(second_root)])

        first_ts = tskit.load(
            self.output_root / "trees" / "sim_poc_easy_0.trees"
        )
        second_ts = tskit.load(second_root / "trees" / "sim_poc_easy_0.trees")
        self.assertTrue(
            first_ts.dump_tables().equals(
                second_ts.dump_tables(), ignore_provenance=True
            )
        )
        first_fasta = self.output_root / "fasta" / "sim_poc_easy_0.fa"
        second_fasta = second_root / "fasta" / "sim_poc_easy_0.fa"
        self.assertEqual(first_fasta.read_bytes(), second_fasta.read_bytes())

    def test_invalid_parameters_are_rejected(self):
        invalid_cases = (
            ({"nrep": 0, "n": 8}, "replicate count"),
            ({"nrep": 1, "n": 7}, "haplotype count"),
            ({"nrep": 1, "n": 8, "length": 0}, "sequence length"),
            ({"nrep": 1, "n": 8, "Ne": 0}, "population size"),
            ({"nrep": 1, "n": 8, "mu": -1}, "rates cannot be negative"),
            ({"nrep": 1, "n": 8, "rec": -1}, "rates cannot be negative"),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    merge_msprime.simulate(pref="invalid", **overrides)


if __name__ == "__main__":
    unittest.main()
