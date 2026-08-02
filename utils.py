import gzip
import math
import os
import pickle
import re
from dataclasses import dataclass

import torch


BASES = ("A", "C", "G", "T")
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
VCF_PARSER_VERSION = "strict-phased-diploid-biallelic-snp-v1"
MAX_SUPPORTED_VCF_BYTES = 1_000_000
MAX_LOCAL_REFINEMENT_SPAN_BP = 100_000.0


@dataclass(frozen=True)
class VCFVariantData:
    input_mode: str
    path: str
    sample_ids: list
    haplotype_ids: list
    sequence_length: int
    positions0: object
    refs: list
    alts: list
    haplotype_partials: object
    parser_version: str

    @property
    def num_haplotypes(self):
        return int(self.haplotype_partials.shape[0])

    @property
    def num_variants(self):
        return int(self.haplotype_partials.shape[1])


def build_scheduler(optimizer, cfg_scheduler):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)
    return scheduler

    
def read_fasta(filepath):
    all_seqs_dict = {}
    with open(filepath, 'r') as file:
        seq_id = None
        all_seqs = []
        for line in file:
            line = line.rstrip()
            if line.startswith('>'):
                if len(all_seqs) > 0 and seq_id is not None:
                    all_seqs_dict[seq_id] = all_seqs
                seq_id = line
                all_seqs = []
            elif len(line) > 0:
                all_seqs.append(line)

        if len(all_seqs) > 0 and seq_id is not None:
            all_seqs_dict[seq_id] = all_seqs

    return all_seqs_dict

def load_sequences(sequences_path):
    if sequences_path.endswith('.fa'):
        key_to_seqs_dict = read_fasta(sequences_path)
        all_seqs = ["".join(lines) for lines in key_to_seqs_dict.values()]
    elif sequences_path.endswith('.pickle'):
        data = pickle.load(open(sequences_path, 'rb'))
        all_seqs = list(data.values()) if isinstance(data, dict) else data
    else:
        all_seqs = pickle.load(open(sequences_path, 'rb'))

    return [seq.replace('?', '-') for seq in all_seqs]


def is_vcf_path(path):
    path = str(path).lower()
    return path.endswith(".vcf") or path.endswith(".vcf.gz")


def load_vcf_variants(vcf_path, max_bytes=MAX_SUPPORTED_VCF_BYTES):
    import numpy as np

    vcf_path = os.path.abspath(str(vcf_path))
    max_bytes = _normalize_optional_positive_int(max_bytes, "max_bytes")
    if max_bytes is not None and not vcf_path.endswith(".gz"):
        _raise_if_file_too_large(vcf_path, max_bytes)

    sample_ids = None
    contig_id = None
    sequence_length = None
    seen_positions = set()
    positions0 = []
    refs = []
    alts = []
    alleles_by_haplotype = []
    uncompressed_bytes = 0

    with _open_text_maybe_gzip(vcf_path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if max_bytes is not None:
                uncompressed_bytes += len(raw_line.encode("utf-8"))
                if uncompressed_bytes > max_bytes:
                    raise ValueError(
                        "VCF input exceeds the supported 1 MB limit "
                        f"({uncompressed_bytes:,} bytes read from {vcf_path}). "
                        "Provide a VCF of at most 1,000,000 bytes, or explicitly "
                        "raise the limit only after profiling memory and runtime."
                    )
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("##contig="):
                parsed_contig_id, parsed_length = _parse_contig_header(line)
                if parsed_contig_id is not None:
                    if contig_id is not None and parsed_contig_id != contig_id:
                        raise ValueError("VCF v1 supports exactly one contig")
                    contig_id = parsed_contig_id
                if parsed_length is not None:
                    sequence_length = parsed_length
                continue
            if line.startswith("#CHROM"):
                fields = line.split("\t")
                if len(fields) < 10:
                    raise ValueError("VCF must contain at least one sample column")
                sample_ids = fields[9:]
                continue
            if line.startswith("#"):
                continue

            if sample_ids is None:
                raise ValueError("VCF records appeared before the #CHROM header")

            fields = line.split("\t")
            if len(fields) < 10:
                raise ValueError(f"VCF record on line {line_number} has no genotype columns")

            chrom, pos_text, _id, ref, alt_text, _qual, _filter, _info, fmt = fields[:9]
            if contig_id is None:
                contig_id = chrom
            elif chrom != contig_id:
                raise ValueError("VCF v1 supports exactly one contig")

            try:
                pos = int(pos_text)
            except ValueError as exc:
                raise ValueError(f"Invalid VCF POS on line {line_number}: {pos_text!r}") from exc
            if pos <= 0:
                raise ValueError(f"VCF POS must be 1-based positive, got {pos}")
            if pos in seen_positions:
                raise ValueError(f"Duplicate VCF position: {pos}")
            seen_positions.add(pos)

            ref = ref.upper()
            alt_values = alt_text.upper().split(",")
            if len(alt_values) != 1:
                raise ValueError(f"Multiallelic record at position {pos} is not supported in VCF v1")
            alt = alt_values[0]
            if len(ref) != 1 or len(alt) != 1 or ref not in BASE_TO_INDEX or alt not in BASE_TO_INDEX:
                raise ValueError(f"Only biallelic A/C/G/T SNPs are supported in VCF v1, got {ref}>{alt} at {pos}")

            format_keys = fmt.split(":")
            try:
                gt_index = format_keys.index("GT")
            except ValueError as exc:
                raise ValueError(f"VCF record at position {pos} is missing FORMAT/GT") from exc

            row_alleles = []
            for sample_id, sample_field in zip(sample_ids, fields[9:]):
                values = sample_field.split(":")
                if gt_index >= len(values):
                    raise ValueError(f"Missing GT for sample {sample_id} at position {pos}")
                gt = values[gt_index]
                if "/" in gt or "|" not in gt:
                    raise ValueError(f"Unphased genotype for sample {sample_id} at position {pos}: {gt}")
                tokens = gt.split("|")
                if len(tokens) != 2:
                    raise ValueError(f"VCF v1 requires diploid GT for sample {sample_id} at position {pos}: {gt}")
                for token in tokens:
                    if token not in {"0", "1"}:
                        raise ValueError(f"Missing or unsupported allele in GT for sample {sample_id} at position {pos}: {gt}")
                    row_alleles.append(ref if token == "0" else alt)

            positions0.append(pos - 1)
            refs.append(ref)
            alts.append(alt)
            alleles_by_haplotype.append(row_alleles)

    if sample_ids is None:
        raise ValueError("VCF header with sample columns was not found")
    if not positions0:
        raise ValueError("VCF contains no supported variant records")
    if sequence_length is None:
        sequence_length = max(positions0) + 1
    if int(sequence_length) <= max(positions0):
        raise ValueError("VCF contig length is shorter than the largest variant position")

    haplotype_ids = [
        f"{sample_id}_h{copy_idx}"
        for sample_id in sample_ids
        for copy_idx in range(2)
    ]
    haplotype_partials = np.zeros(
        (len(haplotype_ids), len(positions0), len(BASES)),
        dtype=np.float32,
    )
    for variant_idx, row_alleles in enumerate(alleles_by_haplotype):
        if len(row_alleles) != len(haplotype_ids):
            raise ValueError("Internal VCF parser error: haplotype count mismatch")
        for hap_idx, allele in enumerate(row_alleles):
            haplotype_partials[hap_idx, variant_idx, BASE_TO_INDEX[allele]] = 1.0

    positions0_array = np.asarray(positions0, dtype=np.int64)
    if np.any(np.diff(positions0_array) <= 0):
        raise ValueError("VCF variant positions must be strictly increasing")

    return VCFVariantData(
        input_mode="vcf",
        path=vcf_path,
        sample_ids=list(sample_ids),
        haplotype_ids=haplotype_ids,
        sequence_length=int(sequence_length),
        positions0=positions0_array,
        refs=refs,
        alts=alts,
        haplotype_partials=haplotype_partials,
        parser_version=VCF_PARSER_VERSION,
    )


def validate_local_refinement_span(
    genomic_range,
    *,
    field_name="genomic_range",
    max_span_bp=MAX_LOCAL_REFINEMENT_SPAN_BP,
):
    if not isinstance(genomic_range, (list, tuple)) or len(genomic_range) != 2:
        raise ValueError(f"{field_name} must be a two-value half-open range")
    left, right = (float(value) for value in genomic_range)
    if (
        not math.isfinite(left)
        or not math.isfinite(right)
        or left < 0.0
        or right <= left
    ):
        raise ValueError(f"{field_name} must satisfy 0 <= left < right with finite coordinates")
    span = float(right - left)
    if span > float(max_span_bp):
        raise ValueError(
            f"{field_name} spans {span:g} bp, which exceeds the supported "
            f"local refinement limit of {float(max_span_bp):g} bp. Split the "
            "request or explicitly raise the limit only after profiling memory "
            "and runtime."
        )
    return span


def _normalize_optional_positive_int(value, field_name):
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer or None")
    return parsed


def _raise_if_file_too_large(path, max_bytes):
    size = os.path.getsize(path)
    if size > int(max_bytes):
        raise ValueError(
            "VCF input exceeds the supported 1 MB limit "
            f"({size:,} bytes on disk at {path}). Provide a VCF of at most "
            "1,000,000 bytes, or explicitly raise the limit only after "
            "profiling memory and runtime."
        )


def _open_text_maybe_gzip(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _parse_contig_header(line):
    body = line.removeprefix("##contig=<").removesuffix(">")
    fields = {}
    for item in body.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    contig_id = fields.get("ID")
    length = fields.get("length")
    if length is None:
        length = fields.get("Length")
    if length is not None and re.fullmatch(r"\d+", length):
        length = int(length)
    else:
        length = None
    return contig_id, length
