"""Lossless observations from the standalone infinite-sites simulator.

Physical coordinates and observation indices are deliberately independent.
This module reads no ancestry, mutation placements, or ground-truth trees.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class SNPData:
    """Polarized, fully observed SNPs; rows are haplotypes, columns are sites.

    Arrays are copied and made read-only on construction. Nucleotide labels
    are retained for provenance; likelihood calculations use the 0/1 patterns.
    """

    genotypes: np.ndarray
    positions: np.ndarray
    sequence_length: float
    site_ids: tuple[int, ...]
    ancestral_states: tuple[str, ...]
    derived_states: tuple[str, ...]
    haplotype_ids: tuple[str, ...]
    contig_id: str = "1"

    def __post_init__(self):
        try:
            length = float(self.sequence_length)
            positions = np.array(self.positions, dtype=np.float64, copy=True)
            genotypes = np.asarray(self.genotypes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid SNP arrays or sequence_length") from exc
        if isinstance(self.sequence_length, (bool, np.bool_)) or not math.isfinite(length) or length <= 0:
            raise ValueError("sequence_length must be finite and positive")
        if positions.ndim != 1 or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite one-dimensional array")
        if np.any(positions < 0) or np.any(positions >= length):
            raise ValueError("positions must lie within [0, sequence_length)")
        if np.any(np.diff(positions) <= 0):
            raise ValueError("positions must be unique and strictly increasing")
        if genotypes.ndim != 2 or genotypes.shape[0] < 2 or genotypes.shape[1] != len(positions):
            raise ValueError("genotypes must have shape [at least 2 haplotypes, num_variants]")
        if not np.all((genotypes == 0) | (genotypes == 1)):
            raise ValueError("genotypes must contain only complete binary 0/1 calls")
        genotypes = np.array(genotypes, dtype=np.uint8, copy=True)
        counts = genotypes.sum(axis=0)
        if np.any((counts == 0) | (counts == genotypes.shape[0])):
            raise ValueError("every recorded SNP must be segregating")

        site_ids = tuple(self.site_ids)
        if (len(site_ids) != len(positions)
                or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                       or x < 0 for x in site_ids)
                or len(set(site_ids)) != len(site_ids)):
            raise ValueError("site_ids must be unique nonnegative integers, one per variant")
        ancestral, derived = tuple(self.ancestral_states), tuple(self.derived_states)
        if (len(ancestral) != len(positions) or len(derived) != len(positions)
                or any(a not in ("A", "C", "G", "T") or d not in ("A", "C", "G", "T")
                       or a == d for a, d in zip(ancestral, derived))):
            raise ValueError("each SNP must have distinct ancestral and derived A/C/G/T labels")
        haplotypes = tuple(self.haplotype_ids)
        if (len(haplotypes) != genotypes.shape[0]
                or any(not isinstance(x, str) or not x for x in haplotypes)
                or len(set(haplotypes)) != len(haplotypes)):
            raise ValueError("haplotype_ids must be unique nonempty strings, one per row")
        if not isinstance(self.contig_id, str) or not self.contig_id:
            raise ValueError("contig_id must be a nonempty string")
        genotypes.setflags(write=False)
        positions.setflags(write=False)
        for name, value in (("genotypes", genotypes), ("positions", positions),
                            ("sequence_length", length), ("site_ids", tuple(map(int, site_ids))),
                            ("ancestral_states", ancestral), ("derived_states", derived),
                            ("haplotype_ids", haplotypes)):
            object.__setattr__(self, name, value)

    @property
    def num_haplotypes(self):
        return self.genotypes.shape[0]

    @property
    def num_variants(self):
        return self.genotypes.shape[1]


def _integer(value, description, minimum=0):
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError
        result = int(value)
        if result < minimum:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{description} must be an integer >= {minimum}") from None
    return result


def _read_position_map(path):
    mapping = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"site_id", "position_zero_based", "vcf_position_one_based"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("position map requires site_id, position_zero_based, vcf_position_one_based")
        for row in reader:
            site_id = _integer(row["site_id"], "position-map site_id")
            if site_id in mapping:
                raise ValueError(f"duplicate position-map site_id {site_id}")
            try:
                position = float(row["position_zero_based"])
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"invalid exact position for site {site_id}") from None
            vcf_position = _integer(row["vcf_position_one_based"], "mapped VCF position", 1)
            mapping[site_id] = (position, vcf_position)
    return mapping


def _read_vcf(path, contig_id, mapping):
    names, records, seen_ids, declared_contigs = None, [], set(), set()
    previous_position = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip("\r\n")
            if line.startswith("##contig=<"):
                attributes = dict(part.split("=", 1) for part in line[len("##contig=<"):].rstrip(">").split(",")
                                  if "=" in part)
                declared_contigs.add(attributes.get("ID"))
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.split("\t")
                if names is not None or fields[:9] != ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]:
                    raise ValueError("invalid or repeated VCF header")
                names = fields[9:]
                if not names or any(not name for name in names) or len(set(names)) != len(names):
                    raise ValueError("VCF requires unique nonempty sample names")
                continue
            if not line or line.startswith("#"):
                raise ValueError(f"unexpected VCF line {line_number}")
            if names is None:
                raise ValueError("VCF records must follow a #CHROM header")
            fields = line.split("\t")
            if len(fields) != 9 + len(names):
                raise ValueError(f"VCF line {line_number} has the wrong number of sample columns")
            if fields[0] != contig_id:
                raise ValueError("VCF must contain only the contig declared in metadata")
            site_id = _integer(fields[2], "VCF site ID")
            position = _integer(fields[1], "VCF position", 1)
            if site_id in seen_ids:
                raise ValueError(f"duplicate VCF site ID {site_id}")
            if position <= previous_position:
                raise ValueError("exported VCF positions must be strictly increasing")
            previous_position = position
            seen_ids.add(site_id)
            if site_id not in mapping or mapping[site_id][1] != position:
                raise ValueError(f"VCF/position-map mismatch at site {site_id}")
            ref, alt = fields[3:5]
            if ref not in ("A", "C", "G", "T") or alt not in ("A", "C", "G", "T") or ref == alt:
                raise ValueError(f"site {site_id} must be a biallelic nucleotide SNP")
            if fields[6] not in ("PASS", "."):
                raise ValueError(f"filtered VCF record at site {site_id}; complete observations required")
            formats = fields[8].split(":")
            if formats.count("GT") != 1:
                raise ValueError(f"site {site_id} requires exactly one GT field")
            gt_index, calls = formats.index("GT"), []
            for name, sample in zip(names, fields[9:]):
                values = sample.split(":")
                alleles = values[gt_index].split("|") if len(values) > gt_index else []
                if len(alleles) != 2 or any(allele not in ("0", "1") for allele in alleles):
                    raise ValueError(f"site {site_id}, sample {name}: complete phased diploid 0|1 calls required")
                calls.extend(map(int, alleles))
            records.append((site_id, mapping[site_id][0], ref, alt, calls))
    if names is None:
        raise ValueError("VCF requires a #CHROM header")
    if declared_contigs != {contig_id}:
        raise ValueError("VCF must declare exactly the metadata contig")
    if seen_ids != mapping.keys():
        raise ValueError("VCF and position map must contain exactly the same site IDs")
    return names, records


def load_snp_dataset(replicate_dir) -> SNPData:
    """Read a schema-v1 simulator bundle, without opening its ground-truth files.

    REF is ancestral only because the simulator explicitly guarantees it.
    General VCFs, missing observations and unknown polarization are unsupported.
    VCF POS and contig length are export coordinates, never physical geometry.
    """
    directory = Path(replicate_dir).expanduser()
    with (directory / "metadata.json").open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("expected simulator metadata schema_version 1")
    if (metadata.get("mutation_model") != "infinite_sites"
            or metadata.get("mutation_positions") != "continuous"
            or metadata.get("mutation_alphabet") != "ACGT"
            or metadata.get("vcf_ref") != "known simulated ancestral allele; no external reference genome"):
        raise ValueError("metadata must declare continuous infinite-sites ACGT data with known ancestral REF")
    parameters, files, summary = (metadata.get(key) for key in ("parameters", "files", "summary"))
    if not all(isinstance(value, dict) for value in (parameters, files, summary)):
        raise ValueError("metadata requires parameters, files, and summary objects")
    try:
        length = parameters["sequence_length"]
        contig_id = str(parameters["contig_id"])
        paths = [files[key] for key in ("vcf", "position_map")]
    except KeyError as exc:
        raise ValueError(f"missing required metadata field {exc.args[0]}") from None
    if any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("metadata VCF and position-map paths must be nonempty strings")
    mapping = _read_position_map(directory / paths[1])
    names, records = _read_vcf(directory / paths[0], contig_id, mapping)
    genotypes = (np.array([row[4] for row in records], dtype=np.uint8).T
                 if records else np.empty((2 * len(names), 0), dtype=np.uint8))
    data = SNPData(
        genotypes=genotypes, positions=[row[1] for row in records], sequence_length=length,
        site_ids=tuple(row[0] for row in records), ancestral_states=tuple(row[2] for row in records),
        derived_states=tuple(row[3] for row in records),
        haplotype_ids=tuple(f"{name}:{copy}" for name in names for copy in range(2)), contig_id=contig_id,
    )
    for source, key, expected in (
        (metadata, "num_sites", data.num_variants), (metadata, "num_mutations", data.num_variants),
        (metadata, "num_haplotypes", data.num_haplotypes), (parameters, "num_samples", data.num_haplotypes),
        (summary, "num_sites", data.num_variants), (summary, "num_mutations", data.num_variants),
        (summary, "num_haplotypes", data.num_haplotypes), (summary, "num_individuals", len(names)),
    ):
        if _integer(source.get(key), f"metadata {key}") != expected:
            raise ValueError(f"metadata {key} does not match observations ({expected})")
    if summary.get("sequence_length_bp") != data.sequence_length:
        raise ValueError("metadata sequence_length_bp disagrees with physical sequence_length")
    return data
