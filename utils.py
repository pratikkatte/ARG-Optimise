from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Sequence, Any
import torch
import math
import json
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

TimedTree = Any

@dataclass(frozen=True)
class ThreadChoice:
    site: int
    branch_child: int
    branch_signature: Tuple[int, ...]
    time_idx: int
    time_value: float
    is_root_branch: bool


@dataclass(frozen=True)
class ThreadPathState:
    site_index: int
    current_choice: Optional[ThreadChoice]

@dataclass(frozen=True)
class MultiLeafState:
    current_graph_segments: Tuple["GraphSegment", ...]
    leaves_threaded: Tuple[int, ...]
    current_focal_leaf: Optional[int]
    inner_state: Optional[ThreadPathState]

@dataclass(frozen=True)
class ArgweaverSitesData:
    leaf_names: Tuple[str, ...]
    contig: str
    region_start: int
    region_end: int
    raw_positions: Tuple[int, ...]
    site_positions: Tuple[float, ...]
    fasta_sequences: Tuple[str, ...]
    geno: torch.Tensor

@dataclass(frozen=True)
class ThreadingConfig:
    geno: torch.Tensor
    time_grid: Tuple[float, ...]
    mutation_rate: float
    recomb_rate: float
    reward_temperature: float
    sequence_length: int
    fasta_sequences: Optional[Tuple[str, ...]] = None
    leaf_names: Optional[Tuple[str, ...]] = None
    substitution_model: str = "JC"
    site_positions: Optional[Tuple[float, ...]] = None
    genomic_region: Optional[Tuple[float, float]] = None

    @classmethod
    def from_raw(
        cls,
        geno,
        time_grid,
        mutation_rate,
        recomb_rate,
        reward_temperature=1.0,
        fasta_path=None,
        fasta_sequences: Optional[Sequence[str]] = None,
        leaf_names: Optional[Sequence[str]] = None,
        substitution_model="JC",
        site_positions: Optional[Sequence[float]] = None,
        genomic_region: Optional[Tuple[float, float]] = None,
    ):
        geno_t = torch.as_tensor(geno, dtype=torch.long)
        substitution_model = str(substitution_model).upper()
        if substitution_model != "JC":
            raise ValueError(
                "Only the JC substitution model is currently supported for FASTA rewards."
            )

        ordered_sequences = None
        ordered_leaf_names = tuple(str(name) for name in leaf_names) if leaf_names is not None else None
        if fasta_path is not None and fasta_sequences is not None:
            raise ValueError("Pass either fasta_path or fasta_sequences, not both")
        if fasta_path is not None:
            ordered_sequences, ordered_leaf_names = _load_fasta_by_leaf_names(
                fasta_path,
                leaf_names=ordered_leaf_names,
                expected_count=int(geno_t.shape[0]),
                allow_order_fallback=True,
            )
        elif fasta_sequences is not None:
            ordered_sequences = tuple(str(seq).upper() for seq in fasta_sequences)

        if ordered_sequences is not None:
            _validate_ordered_fasta_sequences(
                ordered_sequences,
                expected_count=int(geno_t.shape[0]),
                expected_length=int(geno_t.shape[1]),
            )

        ordered_site_positions = _validate_site_positions(
            site_positions,
            expected_length=int(geno_t.shape[1]),
        )
        ordered_genomic_region = _validate_genomic_region(
            genomic_region,
            site_positions=ordered_site_positions,
            sequence_length=int(geno_t.shape[1]),
        )

        return cls(
            geno=geno_t,
            time_grid=tuple(float(t) for t in time_grid),
            mutation_rate=float(mutation_rate),
            recomb_rate=float(recomb_rate),
            reward_temperature=float(reward_temperature),
            sequence_length=int(geno_t.shape[1]),
            fasta_sequences=ordered_sequences,
            leaf_names=ordered_leaf_names,
            substitution_model=substitution_model,
            site_positions=ordered_site_positions,
            genomic_region=ordered_genomic_region,
        )

    @classmethod
    def from_fasta(
        cls,
        fasta_path,
        leaf_names: Sequence[str],
        time_grid,
        mutation_rate,
        recomb_rate,
        reward_temperature=1.0,
        substitution_model="JC",
    ):
        substitution_model = str(substitution_model).upper()
        if substitution_model != "JC":
            raise ValueError(
                "Only the JC substitution model is currently supported for FASTA rewards."
            )

        ordered_sequences, ordered_leaf_names = _load_fasta_by_leaf_names(
            fasta_path,
            leaf_names=leaf_names,
            allow_order_fallback=False,
        )
        geno_t = _numeric_geno_from_fasta_sequences(ordered_sequences)
        return cls(
            geno=geno_t,
            time_grid=tuple(float(t) for t in time_grid),
            mutation_rate=float(mutation_rate),
            recomb_rate=float(recomb_rate),
            reward_temperature=float(reward_temperature),
            sequence_length=int(geno_t.shape[1]),
            fasta_sequences=tuple(ordered_sequences),
            leaf_names=ordered_leaf_names,
            substitution_model=substitution_model,
        )

    @classmethod
    def from_argweaver_sites(
        cls,
        sites_path,
        time_grid,
        mutation_rate,
        recomb_rate,
        reward_temperature=1.0,
        substitution_model="JC",
    ):
        sites = parse_argweaver_sites(sites_path)
        return cls.from_raw(
            sites.geno,
            time_grid,
            mutation_rate=mutation_rate,
            recomb_rate=recomb_rate,
            reward_temperature=reward_temperature,
            fasta_sequences=sites.fasta_sequences,
            leaf_names=sites.leaf_names,
            substitution_model=substitution_model,
            site_positions=sites.site_positions,
            genomic_region=(0.0, float(sites.region_end - sites.region_start + 1)),
        )

    def site_distance(self, site_index: int) -> float:
        if not (0 <= site_index < self.sequence_length):
            raise IndexError(
                f"site_index {site_index} is out of bounds for sequence length {self.sequence_length}"
            )
        if site_index == 0 or self.site_positions is None:
            return 1.0
        return max(float(self.site_positions[site_index] - self.site_positions[site_index - 1]), 1e-12)

    def genomic_interval_for_site_span(self, start: int, end: int) -> Tuple[float, float]:
        if not (0 <= start <= end <= self.sequence_length):
            raise ValueError(
                f"Invalid site span ({start}, {end}) for sequence length {self.sequence_length}"
            )
        if self.site_positions is None or self.genomic_region is None:
            return (float(start), float(end))
        region_start, region_end = self.genomic_region
        left = float(region_start) if start == 0 else float(self.site_positions[start])
        right = float(region_end) if end == self.sequence_length else float(self.site_positions[end])
        if right < left:
            raise ValueError(f"Invalid genomic interval ({left}, {right}) for site span ({start}, {end})")
        return (left, right)


_FASTA_ALLOWED_CODES = frozenset("AGCT-?NRYSWKMBDHUV.")
_FASTA_NUMERIC_CODES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def _validate_site_positions(
    site_positions: Optional[Sequence[float]],
    expected_length: int,
) -> Optional[Tuple[float, ...]]:
    if site_positions is None:
        return None
    ordered = tuple(float(position) for position in site_positions)
    if len(ordered) != int(expected_length):
        raise ValueError(
            f"site_positions length {len(ordered)} != expected {int(expected_length)}"
        )
    for idx, position in enumerate(ordered):
        if not math.isfinite(position):
            raise ValueError(f"site_positions[{idx}] is not finite: {position!r}")
        if idx > 0 and not (position > ordered[idx - 1]):
            raise ValueError("site_positions must be strictly increasing")
    return ordered


def _validate_genomic_region(
    genomic_region: Optional[Tuple[float, float]],
    *,
    site_positions: Optional[Tuple[float, ...]],
    sequence_length: int,
) -> Optional[Tuple[float, float]]:
    if genomic_region is None:
        return None
    if len(genomic_region) != 2:
        raise ValueError("genomic_region must be a (start, end) tuple")
    start, end = (float(genomic_region[0]), float(genomic_region[1]))
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        raise ValueError(f"Invalid genomic_region {genomic_region!r}")
    if site_positions is not None and sequence_length > 0:
        first = site_positions[0]
        last = site_positions[-1]
        if not (start <= first < end and start <= last < end):
            raise ValueError(
                "site_positions must fall inside genomic_region; "
                f"got first={first}, last={last}, region=({start}, {end})"
            )
    return (start, end)


def parse_argweaver_sites(sites_path) -> ArgweaverSitesData:
    """Parse ARGweaver ``.sites`` input into variant-only per-sample columns."""
    leaf_names: Optional[Tuple[str, ...]] = None
    contig: Optional[str] = None
    region_start: Optional[int] = None
    region_end: Optional[int] = None
    positions: list[int] = []
    columns: list[str] = []

    with open(sites_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            tag = fields[0]
            if tag == "NAMES":
                if leaf_names is not None:
                    raise ValueError(f"Duplicate NAMES line in {sites_path}")
                if len(fields) < 2:
                    raise ValueError("ARGweaver .sites NAMES line has no samples")
                leaf_names = tuple(fields[1:])
                if len(set(leaf_names)) != len(leaf_names):
                    raise ValueError("ARGweaver .sites sample names must be unique")
                continue
            if tag == "REGION":
                if contig is not None:
                    raise ValueError(f"Duplicate REGION line in {sites_path}")
                if len(fields) != 4:
                    raise ValueError(
                        f"ARGweaver .sites REGION line must have 4 fields, got {len(fields)}"
                    )
                contig = fields[1]
                try:
                    region_start = int(fields[2])
                    region_end = int(fields[3])
                except ValueError as exc:
                    raise ValueError("ARGweaver .sites REGION bounds must be integers") from exc
                if region_start > region_end:
                    raise ValueError(
                        f"ARGweaver .sites REGION start {region_start} exceeds end {region_end}"
                    )
                continue

            if leaf_names is None or contig is None or region_start is None or region_end is None:
                raise ValueError(
                    f"Variant row appeared before NAMES and REGION at line {line_number}"
                )
            if len(fields) != 2:
                raise ValueError(
                    f"ARGweaver .sites variant line {line_number} must have position and allele string"
                )
            try:
                position = int(fields[0])
            except ValueError as exc:
                raise ValueError(f"Invalid ARGweaver .sites position at line {line_number}") from exc
            alleles = fields[1].upper()
            if len(alleles) != len(leaf_names):
                raise ValueError(
                    f"Allele string width {len(alleles)} at line {line_number} "
                    f"does not match {len(leaf_names)} samples"
                )
            bad = sorted(set(alleles) - _FASTA_ALLOWED_CODES)
            if bad:
                raise ValueError(
                    f"ARGweaver .sites line {line_number} contains unsupported symbols {bad}"
                )
            if position < region_start or position > region_end:
                raise ValueError(
                    f"ARGweaver .sites position {position} lies outside REGION "
                    f"{region_start}-{region_end}"
                )
            if positions and position <= positions[-1]:
                raise ValueError("ARGweaver .sites positions must be strictly increasing")
            positions.append(position)
            columns.append(alleles)

    if leaf_names is None:
        raise ValueError(f"ARGweaver .sites file has no NAMES line: {sites_path}")
    if contig is None or region_start is None or region_end is None:
        raise ValueError(f"ARGweaver .sites file has no REGION line: {sites_path}")
    if not positions:
        raise ValueError(f"ARGweaver .sites file has no variant rows: {sites_path}")

    sequences = tuple(
        "".join(column[sample_idx] for column in columns)
        for sample_idx in range(len(leaf_names))
    )
    geno = _numeric_geno_from_fasta_sequences(sequences)
    site_positions = tuple(float(position - region_start) for position in positions)
    return ArgweaverSitesData(
        leaf_names=leaf_names,
        contig=contig,
        region_start=region_start,
        region_end=region_end,
        raw_positions=tuple(positions),
        site_positions=site_positions,
        fasta_sequences=sequences,
        geno=geno,
    )


def _parse_fasta_records(fasta_path) -> Dict[str, str]:
    records: Dict[str, str] = {}
    current_name: Optional[str] = None
    current_parts: list[str] = []

    def flush_record() -> None:
        nonlocal current_name, current_parts
        if current_name is None:
            return
        sequence = "".join(current_parts).upper()
        if not sequence:
            raise ValueError(f"FASTA record {current_name!r} has no sequence")
        if current_name in records:
            raise ValueError(f"Duplicate FASTA record ID {current_name!r}")
        bad = sorted(set(sequence) - _FASTA_ALLOWED_CODES)
        if bad:
            raise ValueError(
                f"FASTA record {current_name!r} contains unsupported symbols {bad}"
            )
        records[current_name] = sequence
        current_name = None
        current_parts = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                if not header:
                    raise ValueError("FASTA record header is empty")
                current_name = header.split()[0]
                current_parts = []
            else:
                if current_name is None:
                    raise ValueError("FASTA sequence data appeared before any header")
                current_parts.append(line)
    flush_record()

    if not records:
        raise ValueError("FASTA file contains no records")
    return records


def _load_fasta_by_leaf_names(
    fasta_path,
    leaf_names: Optional[Sequence[str]] = None,
    expected_count: Optional[int] = None,
    allow_order_fallback: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    records = _parse_fasta_records(fasta_path)
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(
            f"FASTA has {len(records)} records, expected {int(expected_count)}"
        )

    record_names = tuple(records.keys())
    if not leaf_names:
        ordered_sequences = tuple(records[name] for name in record_names)
        _validate_ordered_fasta_sequences(
            ordered_sequences,
            expected_count=expected_count,
        )
        return ordered_sequences, record_names

    ordered_sequences: list[str] = []
    missing: list[str] = []
    for leaf_name in leaf_names:
        key = str(leaf_name)
        if key not in records:
            missing.append(key)
        else:
            ordered_sequences.append(records[key])

    if missing:
        if allow_order_fallback and len(records) == len(leaf_names):
            ordered_sequences = [records[name] for name in record_names]
            _validate_ordered_fasta_sequences(
                ordered_sequences,
                expected_count=expected_count,
            )
            return tuple(ordered_sequences), record_names
        raise ValueError(f"FASTA is missing records for leaf_names {missing}")

    _validate_ordered_fasta_sequences(
        ordered_sequences,
        expected_count=expected_count,
    )
    return tuple(ordered_sequences), tuple(str(name) for name in leaf_names)


def _validate_ordered_fasta_sequences(
    ordered_sequences: Sequence[str],
    expected_count: Optional[int] = None,
    expected_length: Optional[int] = None,
) -> None:
    if expected_count is not None and len(ordered_sequences) != int(expected_count):
        raise ValueError(
            f"FASTA sequence count {len(ordered_sequences)} != expected {int(expected_count)}"
        )
    for idx, sequence in enumerate(ordered_sequences):
        bad = sorted(set(str(sequence).upper()) - _FASTA_ALLOWED_CODES)
        if bad:
            raise ValueError(f"FASTA sequence {idx} contains unsupported symbols {bad}")
    lengths = {len(seq) for seq in ordered_sequences}
    if len(lengths) != 1:
        raise ValueError(
            "All FASTA sequences mapped by leaf_names must have the same length; "
            f"found lengths {sorted(lengths)}"
        )
    if next(iter(lengths)) == 0:
        raise ValueError("FASTA sequences must not be empty")
    if expected_length is not None and next(iter(lengths)) != int(expected_length):
        raise ValueError(
            f"FASTA sequence length {next(iter(lengths))} != expected {int(expected_length)}"
        )


def _numeric_geno_from_fasta_sequences(sequences: Sequence[str]) -> torch.Tensor:
    encoded = [
        [_FASTA_NUMERIC_CODES.get(base, -1) for base in sequence.upper()]
        for sequence in sequences
    ]
    return torch.as_tensor(encoded, dtype=torch.long)

@dataclass(frozen=True)
class GraphSegment:
    start: int
    end: int
    graph: Data
    genomic_interval: Optional[Tuple[float, float]] = None

    @property
    def edge_index(self) -> torch.Tensor:
        return self.graph.edge_index

    @property
    def num_nodes(self) -> int:
        return int(self.graph.num_nodes)

    @property
    def root(self) -> int:
        return graph_root(self.graph)

    @property
    def node_times(self) -> torch.Tensor:
        return self.graph.node_times

    @property
    def leaf_ids(self) -> Tuple[int, ...]:
        return tuple(
            sorted(sample_id for sample_id in graph_node_sample_ids(self.graph) if sample_id >= 0)
        )

    @property
    def node_sample_ids(self) -> Tuple[int, ...]:
        return graph_node_sample_ids(self.graph)


BackboneSegment = GraphSegment

@dataclass(frozen=True)
class SiteBackboneTree:
    site: int
    edge_index: torch.Tensor
    num_nodes: int
    root: int
    node_times: torch.Tensor
    branch_children: Tuple[int, ...]
    branch_signatures: Tuple[Tuple[int, ...], ...]
    parent_of_child: Tuple[int, ...]
    node_sample_ids: Tuple[int, ...]
    descendant_signatures: Tuple[Tuple[int, ...], ...]


def _as_1d_long_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long).reshape(-1).clone()


def _as_1d_float_tensor(values: Sequence[float] | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1).clone()


def _scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def graph_root(graph: Data) -> int:
    return _scalar_int(graph.root)


def graph_site(graph: Data) -> int:
    return _scalar_int(getattr(graph, "site", torch.tensor([-1], dtype=torch.long)))


def graph_node_sample_ids(graph: Data) -> Tuple[int, ...]:
    return tuple(int(x) for x in graph.node_sample_ids.detach().cpu().reshape(-1).tolist())


def graph_parent_of_child(graph: Data) -> Tuple[int, ...]:
    parent = getattr(graph, "parent_of_child", None)
    if parent is not None:
        return tuple(int(x) for x in parent.detach().cpu().reshape(-1).tolist())
    return _parent_from_edge_index(graph.edge_index, int(graph.num_nodes))


def graph_children(graph: Data) -> Tuple[Tuple[int, ...], ...]:
    return _children_from_edge_index(graph.edge_index, int(graph.num_nodes))


def graph_branch_children(graph: Data) -> Tuple[int, ...]:
    branch_children = getattr(graph, "branch_children", None)
    if branch_children is not None:
        return tuple(int(x) for x in branch_children.detach().cpu().reshape(-1).tolist())
    parent = graph_parent_of_child(graph)
    return tuple(node for node in range(int(graph.num_nodes)) if parent[node] != -1)


def graph_branch_signatures(graph: Data) -> Tuple[Tuple[int, ...], ...]:
    signatures = getattr(graph, "branch_signatures", None)
    if signatures is not None:
        return tuple(tuple(int(x) for x in sig) for sig in signatures)
    descendants = graph_descendant_signatures(graph)
    return tuple(descendants[node] for node in graph_branch_children(graph))


def graph_descendant_signatures(graph: Data) -> Tuple[Tuple[int, ...], ...]:
    signatures = getattr(graph, "descendant_signatures", None)
    if signatures is not None:
        return tuple(tuple(int(x) for x in sig) for sig in signatures)

    children = graph_children(graph)
    node_sample_ids = graph_node_sample_ids(graph)

    def descendants(node: int) -> tuple[int, ...]:
        sample_id = node_sample_ids[node]
        if sample_id >= 0:
            return (sample_id,)
        leaf_ids: list[int] = []
        for child in children[node]:
            leaf_ids.extend(descendants(child))
        return tuple(sorted(leaf_ids))

    return tuple(descendants(node) for node in range(int(graph.num_nodes)))


def _graph_x_from_metadata(
    node_times: torch.Tensor,
    node_sample_ids: Sequence[int],
    num_samples: int,
) -> torch.Tensor:
    node_times = node_times.float().reshape(-1)
    node_sample_ids_t = torch.as_tensor(node_sample_ids, dtype=torch.long).reshape(-1)
    x = torch.zeros((node_times.numel(), int(num_samples) + 2), dtype=torch.float32)
    for node_id, sample_id in enumerate(node_sample_ids_t.tolist()):
        if sample_id >= 0:
            x[node_id, int(sample_id)] = 1.0
            x[node_id, int(num_samples) + 1] = 1.0
    max_time = torch.clamp(node_times.max(), min=1e-8)
    x[:, int(num_samples)] = node_times / max_time
    return x


def refresh_graph_metadata(graph: Data, *, num_samples: Optional[int] = None) -> Data:
    graph.edge_index = graph.edge_index.long().contiguous()
    graph.node_times = graph.node_times.float().reshape(-1)
    graph.node_sample_ids = _as_1d_long_tensor(graph.node_sample_ids)
    graph.root = torch.tensor([graph_root(graph)], dtype=torch.long)
    if not hasattr(graph, "site"):
        graph.site = torch.tensor([-1], dtype=torch.long)
    else:
        graph.site = torch.tensor([graph_site(graph)], dtype=torch.long)

    inferred_num_samples = int(
        getattr(graph, "num_samples", torch.tensor([0], dtype=torch.long)).reshape(-1)[0].item()
        if isinstance(getattr(graph, "num_samples", None), torch.Tensor)
        else getattr(graph, "num_samples", 0)
    )
    node_sample_ids = graph_node_sample_ids(graph)
    observed_num_samples = max((sample_id for sample_id in node_sample_ids if sample_id >= 0), default=-1) + 1
    resolved_num_samples = int(num_samples or inferred_num_samples or observed_num_samples)
    graph.num_samples = torch.tensor([resolved_num_samples], dtype=torch.long)
    graph.num_nodes = int(graph.node_times.numel())
    graph.x = _graph_x_from_metadata(graph.node_times, node_sample_ids, resolved_num_samples)
    graph.message_edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)

    parent = _parent_from_edge_index(graph.edge_index, graph.num_nodes)
    graph.parent_of_child = torch.tensor(parent, dtype=torch.long)
    descendants = graph_descendant_signatures(graph)
    branch_children = tuple(node for node in range(graph.num_nodes) if parent[node] != -1)
    graph.branch_children = torch.tensor(branch_children, dtype=torch.long)
    graph.descendant_signatures = descendants
    graph.branch_signatures = tuple(descendants[node] for node in branch_children)
    return graph


def make_tree_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    node_times: Sequence[float] | torch.Tensor,
    node_sample_ids: Sequence[int] | torch.Tensor,
    *,
    num_samples: Optional[int] = None,
    site: int = -1,
) -> Data:
    graph = Data(
        edge_index=edge_index.long().contiguous(),
        node_times=_as_1d_float_tensor(node_times),
        node_sample_ids=_as_1d_long_tensor(node_sample_ids),
        root=torch.tensor([int(root)], dtype=torch.long),
        site=torch.tensor([int(site)], dtype=torch.long),
        num_nodes=int(num_nodes),
    )
    return refresh_graph_metadata(graph, num_samples=num_samples)


def graph_with_site(graph: Data, site: int) -> Data:
    out = graph.clone()
    out.site = torch.tensor([int(site)], dtype=torch.long)
    return out


def graph_to_site_backbone_tree(graph: Data) -> SiteBackboneTree:
    return SiteBackboneTree(
        site=graph_site(graph),
        edge_index=graph.edge_index,
        num_nodes=int(graph.num_nodes),
        root=graph_root(graph),
        node_times=graph.node_times,
        branch_children=graph_branch_children(graph),
        branch_signatures=graph_branch_signatures(graph),
        parent_of_child=graph_parent_of_child(graph),
        node_sample_ids=graph_node_sample_ids(graph),
        descendant_signatures=graph_descendant_signatures(graph),
    )


def attachment_candidate_node_ids(site_tree: SiteBackboneTree | Data) -> List[int]:
    """
    Nodes on which a leaf can attach, matching :func:`enumerate_thread_choices` branches:
    every backbone branch child, plus the root (for the root-regrafting action set).
    """
    if isinstance(site_tree, Data):
        cands = set(graph_branch_children(site_tree))
        cands.add(graph_root(site_tree))
    else:
        cands = set(site_tree.branch_children)
        cands.add(site_tree.root)
    return sorted(cands)


def _graph_postorder(root: int, children: Tuple[Tuple[int, ...], ...]) -> List[int]:
    order: List[int] = []

    def visit(u: int) -> None:
        for v in children[u]:
            visit(v)
        order.append(u)

    visit(root)
    return order


def _graph_preorder(root: int, children: Tuple[Tuple[int, ...], ...]) -> List[int]:
    order: List[int] = []

    def visit(u: int) -> None:
        order.append(u)
        for v in children[u]:
            visit(v)

    visit(root)
    return order

def thread_leaf_into_graph(
    site_graph: SiteBackboneTree | Data,
    focal_leaf: int,
    choice: ThreadChoice,
) -> Data:
    if isinstance(site_graph, Data):
        edge_index = site_graph.edge_index
        root_in = graph_root(site_graph)
        parent_of_child = graph_parent_of_child(site_graph)
        node_times_in = site_graph.node_times
        node_sample_ids_in = graph_node_sample_ids(site_graph)
        num_samples = _scalar_int(site_graph.num_samples)
        site = graph_site(site_graph)
    else:
        edge_index = site_graph.edge_index
        root_in = site_graph.root
        parent_of_child = site_graph.parent_of_child
        node_times_in = site_graph.node_times
        node_sample_ids_in = site_graph.node_sample_ids
        num_samples = max(max(node_sample_ids_in, default=-1) + 1, int(focal_leaf) + 1)
        site = site_graph.site

    edges = [tuple(edge) for edge in edge_index.t().tolist()]
    node_times = node_times_in.tolist()
    node_sample_ids = list(node_sample_ids_in)

    focal_node = len(node_times)
    node_times.append(0.0)
    node_sample_ids.append(focal_leaf)

    if choice.is_root_branch:
        new_root = len(node_times)
        node_times.append(float(choice.time_value))
        node_sample_ids.append(-1)
        edges.append((new_root, root_in))
        edges.append((new_root, focal_node))
        root = new_root
    else:
        parent = parent_of_child[choice.branch_child]
        if parent == -1:
            raise ValueError("Non-root branch choice cannot point to the root without is_root_branch")

        new_internal = len(node_times)
        node_times.append(float(choice.time_value))
        node_sample_ids.append(-1)
        edges = [(u, v) for (u, v) in edges if not (u == parent and v == choice.branch_child)]
        edges.extend(
            [
                (parent, new_internal),
                (new_internal, choice.branch_child),
                (new_internal, focal_node),
            ]
        )
        root = root_in

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    out = make_tree_graph(
        edge_index=edge_index,
        num_nodes=len(node_times),
        root=root,
        node_times=torch.tensor(node_times, dtype=torch.float32),
        node_sample_ids=tuple(node_sample_ids),
        num_samples=max(num_samples, int(focal_leaf) + 1),
        site=site,
    )
    out.choice = choice
    return out


_thread_leaf_into_site_tree_full = thread_leaf_into_graph

def _choice_prior_weights(curr_choices: Sequence[ThreadChoice]) -> torch.Tensor:
    if len(curr_choices) == 0:
        raise ValueError("Need at least one current-site choice")
    time_counts: dict[int, int] = {}
    for choice in curr_choices:
        time_counts[choice.time_idx] = time_counts.get(choice.time_idx, 0) + 1

    log_weights = []
    for choice in curr_choices:
        log_weights.append(-float(choice.time_value) - math.log(time_counts[choice.time_idx]))
    log_weight_tensor = torch.tensor(log_weights, dtype=torch.float32)
    return torch.softmax(log_weight_tensor, dim=0)

def timed_tree_from_graph(
    edge_index: torch.Tensor | Data,
    num_nodes: Optional[int] = None,
    root: Optional[int] = None,
    node_times: Optional[torch.Tensor] = None,
    node_sample_ids: Optional[Sequence[int]] = None,
) -> TimedTree:
    if isinstance(edge_index, Data):
        graph = edge_index
        edge_index = graph.edge_index
        num_nodes = int(graph.num_nodes)
        root = graph_root(graph)
        node_times = graph.node_times
        node_sample_ids = graph_node_sample_ids(graph)
    if num_nodes is None or root is None or node_times is None or node_sample_ids is None:
        raise ValueError("timed_tree_from_graph needs either a PyG Data graph or all graph fields")
    children = _children_from_edge_index(edge_index, num_nodes)

    def build(node_id: int) -> TimedTree:
        sample_id = node_sample_ids[node_id]
        if sample_id >= 0:
            return sample_id
        kids = children[node_id]
        if len(kids) != 2:
            raise ValueError(
                f"Internal node {node_id} has {len(kids)} children, expected 2"
            )
        left = build(kids[0])
        right = build(kids[1])
        return ("n", float(node_times[node_id].item()), left, right)

    return build(root)


def graph_to_timed_tree(graph: Data) -> TimedTree:
    return timed_tree_from_graph(graph)


def graph_segments_to_timed_tree_segments(
    segments: Sequence[GraphSegment],
) -> Tuple[Dict[str, Any], ...]:
    converted: list[Dict[str, Any]] = []
    for segment in segments:
        item: Dict[str, Any] = {
            "sites": (int(segment.start), int(segment.end)),
            "tree": graph_to_timed_tree(segment.graph),
        }
        if segment.genomic_interval is not None:
            item["genomic_interval"] = tuple(float(x) for x in segment.genomic_interval)
        converted.append(item)
    return tuple(converted)


def graphs_equivalent(left: Data, right: Data) -> bool:
    return (
        int(left.num_nodes) == int(right.num_nodes)
        and graph_root(left) == graph_root(right)
        and torch.equal(left.edge_index.cpu(), right.edge_index.cpu())
        and torch.equal(left.node_times.cpu(), right.node_times.cpu())
        and graph_node_sample_ids(left) == graph_node_sample_ids(right)
    )

def serialize_timed_tree(tree: Any) -> Any:
    if isinstance(tree, int):
        return int(tree)
    label, time_value, left, right = tree
    return [
        label,
        float(time_value),
        serialize_timed_tree(left),
        serialize_timed_tree(right),
    ]

def write_segments_json(
    path,
    segments: Sequence[dict[str, Any]] | Sequence[GraphSegment],
    *,
    sequence_length: float,
    num_samples: int,
    coordinate_mode: str,
) -> None:
    if segments and isinstance(segments[0], GraphSegment):
        segments = graph_segments_to_timed_tree_segments(segments)  # type: ignore[assignment]
    serial_segments = []
    for segment in segments:
        item = {
            "sites": list(segment["sites"]),
            "tree": serialize_timed_tree(segment["tree"]),
        }
        if "genomic_interval" in segment:
            item["genomic_interval"] = list(segment["genomic_interval"])
        serial_segments.append(item)
    payload = {
        "sequence_length": float(sequence_length),
        "num_samples": int(num_samples),
        "coordinate_mode": str(coordinate_mode),
        "segments": serial_segments,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_graph_segments_json(
    path,
    segments: Sequence[GraphSegment],
    *,
    sequence_length: float,
    num_samples: int,
    coordinate_mode: str,
) -> None:
    write_segments_json(
        path,
        segments,
        sequence_length=sequence_length,
        num_samples=num_samples,
        coordinate_mode=coordinate_mode,
    )

def segments_to_physical_tskit(
    segments: Sequence[dict[str, Any]] | Sequence[GraphSegment],
    *,
    num_samples: int,
    sequence_length: float,
    output_path,
    interval_key: str = "genomic_interval",
) -> None:
    """Write sampled local-tree segments as a physical-coordinate tskit file."""
    if segments and isinstance(segments[0], GraphSegment):
        segments = graph_segments_to_timed_tree_segments(segments)  # type: ignore[assignment]
    try:
        import tskit
    except ImportError as exc:
        raise ImportError("segments_to_physical_tskit requires the tskit package") from exc

    sequence_length = float(sequence_length)
    if not math.isfinite(sequence_length) or sequence_length <= 0.0:
        raise ValueError(f"sequence_length must be positive and finite, got {sequence_length!r}")

    tables = tskit.TableCollection(sequence_length=sequence_length)
    for _sample_id in range(int(num_samples)):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)

    covered_right = 0.0
    time_eps = 1e-6

    def add_subtree(node: Any, left_bound: float, right_bound: float) -> tuple[int, float]:
        if isinstance(node, int):
            sample_id = int(node)
            if not (0 <= sample_id < int(num_samples)):
                raise ValueError(
                    f"Sample id {sample_id} is out of range for {int(num_samples)} samples"
                )
            return sample_id, 0.0
        _label, time_value, left, right = node
        parent_time = float(time_value)
        if not math.isfinite(parent_time):
            raise ValueError(
                f"Encountered non-finite parent time {parent_time!r} in interval ({left_bound}, {right_bound})"
            )
        left_id, left_time = add_subtree(left, left_bound, right_bound)
        right_id, right_time = add_subtree(right, left_bound, right_bound)
        max_child_time = max(left_time, right_time)
        if parent_time <= max_child_time + time_eps:
            if parent_time < max_child_time - 1e-3:
                raise ValueError(
                    "Invalid tree time ordering in interval "
                    f"({left_bound}, {right_bound}): parent time {parent_time} is much smaller "
                    f"than child max time {max_child_time}"
                )
            parent_time = max_child_time + time_eps
        if not (parent_time > left_time and parent_time > right_time):
            raise ValueError(
                "Invalid tree time ordering in interval "
                f"({left_bound}, {right_bound}): parent time {parent_time} must be greater "
                f"than child times ({left_time}, {right_time})"
            )
        parent_id = tables.nodes.add_row(time=parent_time)
        tables.edges.add_row(left=left_bound, right=right_bound, parent=parent_id, child=left_id)
        tables.edges.add_row(left=left_bound, right=right_bound, parent=parent_id, child=right_id)
        return parent_id, parent_time

    for segment in segments:
        if interval_key not in segment:
            raise ValueError(
                f"Segment {segment.get('sites')} is missing required {interval_key!r}"
            )
        left_bound, right_bound = (float(x) for x in segment[interval_key])
        if left_bound < -1e-8 or right_bound > sequence_length + 1e-8 or right_bound <= left_bound:
            raise ValueError(
                f"Invalid physical interval ({left_bound}, {right_bound}) for sequence length {sequence_length}"
            )
        if not math.isclose(left_bound, covered_right, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Segments must continuously cover the sequence; expected left {covered_right}, got {left_bound}"
            )
        add_subtree(segment["tree"], max(0.0, left_bound), min(sequence_length, right_bound))
        covered_right = right_bound

    if not math.isclose(covered_right, sequence_length, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"Segments cover physical length {covered_right}, expected {sequence_length}"
        )

    tables.sort()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tables.tree_sequence().dump(output_path)


def graph_segments_to_physical_tskit(
    segments: Sequence[GraphSegment],
    *,
    num_samples: int,
    sequence_length: float,
    output_path,
    interval_key: str = "genomic_interval",
) -> None:
    segments_to_physical_tskit(
        segments,
        num_samples=num_samples,
        sequence_length=sequence_length,
        output_path=output_path,
        interval_key=interval_key,
    )

def _binary_site_log_likelihood(
    edge_index: torch.Tensor,
    num_nodes: int,
    root: int,
    node_times: torch.Tensor,
    node_sample_ids: Sequence[int],
    site_observation: torch.Tensor,
    theta: float,
) -> float:
    children = _children_from_edge_index(edge_index, num_nodes)
    site_values = site_observation.tolist()

    def postorder(node: int) -> torch.Tensor:
        sample_id = node_sample_ids[node]
        if sample_id >= 0:
            state = int(site_values[sample_id])
            out = torch.full((2,), -torch.inf, dtype=torch.float64)
            out[state] = 0.0
            return out

        node_log_like = torch.zeros(2, dtype=torch.float64)
        for child in children[node]:
            branch_length = float(node_times[node].item() - node_times[child].item())
            exp_term = math.exp(-2.0 * theta * max(branch_length, 1e-6))
            p_same = 0.5 + 0.5 * exp_term
            p_diff = 0.5 - 0.5 * exp_term
            transition = torch.tensor(
                [[p_same, p_diff], [p_diff, p_same]],
                dtype=torch.float64,
            ).clamp_min(torch.finfo(torch.float64).tiny)
            child_like = postorder(child)
            contrib = torch.logsumexp(torch.log(transition) + child_like.unsqueeze(0), dim=1)
            node_log_like = node_log_like + contrib
        return node_log_like

    root_like = postorder(root)
    log_root_prior = torch.full((2,), math.log(0.5), dtype=torch.float64)
    return float(torch.logsumexp(log_root_prior + root_like, dim=0).item())

def compress_thread_path_to_segments(
    thread_path: Sequence[ThreadChoice],
) -> tuple[dict, ...]:
    if not thread_path:
        return tuple()

    segments: list[dict] = []
    start = 0
    current = thread_path[0]
    current_key = _canonical_choice(current)

    for idx in range(1, len(thread_path)):
        next_key = _canonical_choice(thread_path[idx])
        if next_key != current_key:
            segments.append({"start": start, "end": idx, "choice": current})
            start = idx
            current = thread_path[idx]
            current_key = next_key
    segments.append({"start": start, "end": len(thread_path), "choice": current})
    return tuple(segments)

def _canonical_choice(choice: ThreadChoice) -> tuple[int, tuple[int, ...], int, bool]:
    return (
        choice.branch_child,
        choice.branch_signature,
        choice.time_idx,
        choice.is_root_branch,
    )

def transition_log_probs(
    prev_choices: Sequence[ThreadChoice],
    curr_choices: Sequence[ThreadChoice],
    rho: float,
    time_grid: Sequence[float],
    site_distance: int = 1,
) -> torch.Tensor:
    if len(prev_choices) == 0 or len(curr_choices) == 0:
        raise ValueError("Transition matrices need non-empty previous and current choices")

    recomb_mass = 1.0 - math.exp(-rho * site_distance)
    stay_mass = 1.0 - recomb_mass
    curr_prior = _choice_prior_weights(curr_choices)

    matrix = torch.empty((len(prev_choices), len(curr_choices)), dtype=torch.float32)
    for prev_idx, prev_choice in enumerate(prev_choices):
        prev_key = _canonical_choice(prev_choice)
        for curr_idx, curr_choice in enumerate(curr_choices):
            prob = recomb_mass * float(curr_prior[curr_idx].item())
            if prev_key == _canonical_choice(curr_choice):
                prob += stay_mass
            matrix[prev_idx, curr_idx] = math.log(max(prob, 1e-12))
    return matrix

def _parent_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> tuple[int, ...]:
    parent = [-1] * num_nodes
    if edge_index.numel() > 0:
        for u, v in edge_index.t().tolist():
            parent[v] = u
    return tuple(parent)

def _children_from_edge_index(edge_index: torch.Tensor, num_nodes: int) -> tuple[tuple[int, ...], ...]:
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        for u, v in edge_index.t().tolist():
            children[u].append(v)
    return tuple(tuple(child) for child in children)


def unthread_leaf_from_timed_tree(timed_tree: TimedTree, focal_leaf: int) -> TimedTree:
    def remove(node: TimedTree) -> Optional[TimedTree]:
        if isinstance(node, int):
            return None if node == focal_leaf else node

        label, time_value, left, right = node
        left_out = remove(left)
        right_out = remove(right)
        if left_out is None and right_out is None:
            return None
        if left_out is None:
            return right_out
        if right_out is None:
            return left_out
        return (label, time_value, left_out, right_out)

    out = remove(timed_tree)
    if out is None:
        raise ValueError("Unthreading removed the entire tree")
    return out


def unthread_leaf_from_graph(full_graph: Data, focal_leaf: int) -> Data:
    node_sample_ids = list(graph_node_sample_ids(full_graph))
    matches = [idx for idx, sample_id in enumerate(node_sample_ids) if sample_id == int(focal_leaf)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected focal leaf {focal_leaf} exactly once in graph, found {len(matches)}"
        )
    focal_node = matches[0]
    parent = list(graph_parent_of_child(full_graph))
    children = [list(child_set) for child_set in graph_children(full_graph)]
    focal_parent = parent[focal_node]
    if focal_parent == -1:
        raise ValueError("Unthreading removed the root; at least one non-focal leaf is required")

    siblings = [child for child in children[focal_parent] if child != focal_node]
    if len(siblings) != 1:
        raise ValueError(
            f"Focal parent {focal_parent} has {len(siblings) + 1} children; expected binary tree"
        )
    sibling = siblings[0]
    grandparent = parent[focal_parent]
    old_root = graph_root(full_graph)
    remove = {focal_node, focal_parent}
    keep_nodes = [node for node in range(int(full_graph.num_nodes)) if node not in remove]
    id_map = {old: new for new, old in enumerate(keep_nodes)}

    edges: list[tuple[int, int]] = []
    for u, v in full_graph.edge_index.t().tolist():
        if u in remove or v in remove:
            continue
        edges.append((id_map[int(u)], id_map[int(v)]))
    if grandparent != -1:
        edges.append((id_map[grandparent], id_map[sibling]))

    if old_root == focal_parent:
        new_root_old = sibling
    else:
        new_root_old = old_root

    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    node_times = full_graph.node_times[keep_nodes].clone()
    kept_sample_ids = [node_sample_ids[node] for node in keep_nodes]
    return make_tree_graph(
        edge_index=edge_index,
        num_nodes=len(keep_nodes),
        root=id_map[new_root_old],
        node_times=node_times,
        node_sample_ids=kept_sample_ids,
        num_samples=_scalar_int(full_graph.num_samples),
        site=graph_site(full_graph),
    )


def build_backbone_segments_from_graph_segments(
    reference_graph_segments: Sequence[GraphSegment],
    focal_leaf: int,
) -> Tuple[GraphSegment, ...]:
    segments: list[GraphSegment] = []
    for segment in reference_graph_segments:
        backbone_graph = unthread_leaf_from_graph(segment.graph, focal_leaf)
        segments.append(
            GraphSegment(
                start=int(segment.start),
                end=int(segment.end),
                genomic_interval=segment.genomic_interval,
                graph=backbone_graph,
            )
        )
    return tuple(segments)


def build_graph_segments_from_timed_trees(
    reference_full_trees: Sequence[dict],
    *,
    num_samples: Optional[int] = None,
) -> Tuple[GraphSegment, ...]:
    segments: list[GraphSegment] = []
    for segment in reference_full_trees:
        start, end = segment["sites"]
        edge_index, num_nodes, root, node_times, node_sample_ids = _timed_tree_to_graph_full(segment["tree"])
        graph = make_tree_graph(
            edge_index=edge_index,
            num_nodes=num_nodes,
            root=root,
            node_times=node_times,
            node_sample_ids=node_sample_ids,
            num_samples=num_samples,
        )
        segments.append(
            GraphSegment(
                start=int(start),
                end=int(end),
                genomic_interval=tuple(segment["genomic_interval"]) if "genomic_interval" in segment else None,
                graph=graph,
            )
        )
    return tuple(segments)


def build_backbone_segments_from_reference(
    reference_full_trees: Sequence[dict],
    focal_leaf: int,
) -> Tuple[GraphSegment, ...]:
    full_segments = build_graph_segments_from_timed_trees(reference_full_trees)
    return build_backbone_segments_from_graph_segments(full_segments, focal_leaf)


def _timed_tree_to_graph_full(
    timed_tree: TimedTree,
) -> tuple[torch.Tensor, int, int, torch.Tensor, Tuple[int, ...]]:
    node_times: list[float] = []
    node_sample_ids: list[int] = []
    edges: list[tuple[int, int]] = []

    def allocate_node(time_value: float, sample_id: int) -> int:
        node_id = len(node_times)
        node_times.append(float(time_value))
        node_sample_ids.append(sample_id)
        return node_id

    def visit(node: TimedTree) -> int:
        if isinstance(node, int):
            return allocate_node(0.0, node)

        label, time_value, left, right = node
        if label != "n":
            raise ValueError(f"Unexpected timed-tree node label: {label}")

        left_id = visit(left)
        right_id = visit(right)
        node_id = allocate_node(float(time_value), -1)
        edges.append((node_id, left_id))
        edges.append((node_id, right_id))
        return node_id

    root = visit(timed_tree)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return (
        edge_index,
        len(node_times),
        root,
        torch.tensor(node_times, dtype=torch.float32),
        tuple(node_sample_ids),
    )

def expand_backbone_segments(
    backbone_segments: Sequence[BackboneSegment],
    sequence_length: int,
) -> list[SiteBackboneTree]:
    site_trees: list[Optional[SiteBackboneTree]] = [None] * sequence_length
    for segment in backbone_segments:
        parent_of_child = _parent_from_edge_index(segment.edge_index, segment.num_nodes)
        children = _children_from_edge_index(segment.edge_index, segment.num_nodes)

        def descendants(node: int) -> tuple[int, ...]:
            sample_id = segment.node_sample_ids[node]
            if sample_id >= 0:
                return (sample_id,)
            leaf_ids: list[int] = []
            for child in children[node]:
                leaf_ids.extend(descendants(child))
            return tuple(sorted(leaf_ids))

        descendant_signatures = tuple(descendants(node) for node in range(segment.num_nodes))
        branch_children = tuple(node for node in range(segment.num_nodes) if parent_of_child[node] != -1)
        branch_signatures = tuple(descendant_signatures[node] for node in branch_children)

        site_tree = SiteBackboneTree(
            site=-1,
            edge_index=segment.edge_index,
            num_nodes=segment.num_nodes,
            root=segment.root,
            node_times=segment.node_times,
            branch_children=branch_children,
            branch_signatures=branch_signatures,
            parent_of_child=parent_of_child,
            node_sample_ids=segment.node_sample_ids,
            descendant_signatures=descendant_signatures,
        )
        for site in range(segment.start, min(segment.end, sequence_length)):
            site_trees[site] = SiteBackboneTree(
                site=site,
                edge_index=site_tree.edge_index,
                num_nodes=site_tree.num_nodes,
                root=site_tree.root,
                node_times=site_tree.node_times,
                branch_children=site_tree.branch_children,
                branch_signatures=site_tree.branch_signatures,
                parent_of_child=site_tree.parent_of_child,
                node_sample_ids=site_tree.node_sample_ids,
                descendant_signatures=site_tree.descendant_signatures,
            )

    missing = [idx for idx, site_tree in enumerate(site_trees) if site_tree is None]
    if missing:
        raise ValueError(f"Backbone segments do not cover sites: {missing}")
    return [site_tree for site_tree in site_trees if site_tree is not None]

def get_max_actions(num_backbone_lineages: int, time_grid: Sequence[float]) -> int:
    """
    Calculates the absolute theoretical upper bound of mathematical actions 
    if no lineages ever coalesced until the final time point.
    """
    num_grid_points = sum(1 for t in time_grid if t > 0)
    return num_backbone_lineages * num_grid_points

def get_mathematically_valid_actions(backbone_edges: list[dict], time_grid: Sequence[float]) -> list[tuple]:
    """
    Iterates through the actual backbone_edges.
    Finds every valid intersection where a time point t from the time_grid falls strictly 
    within the edge's lifespan: start_time < t <= end_time.
    Returns a list of all valid (edge_id, t) tuples.
    """
    valid_actions = []
    for edge in backbone_edges:
        edge_id = edge["edge_id"]
        start_time = edge["start_time"]
        end_time = edge["end_time"]
        for t in time_grid:
            if start_time < t <= end_time:
                valid_actions.append((edge_id, t))
    return valid_actions

def get_distinct_topological_outcomes(mathematical_actions: list[tuple], backbone_edges: list[dict]) -> list[tuple]:
    """
    Takes the mathematical actions and collapses them based on topological equivalence.
    Logic: If two or more edges share the exact same end_time (e.g., t = 1.0) AND 
    merge into the exact same target_node_at_end, then regrafting the pruned lineage 
    onto *any* of those edges at t = 1.0 results in the exact same polytomy.
    
    Returns a list of distinct actions (states/choices).
    """
    edge_id_to_target = {e["edge_id"]: e["target_node_at_end"] for e in backbone_edges}
    edge_id_to_end_time = {e["edge_id"]: e["end_time"] for e in backbone_edges}
    
    seen_polytomies = set()
    distinct_actions = []
    
    for edge_id, t in mathematical_actions:
        end_time = edge_id_to_end_time[edge_id]
        target_node = edge_id_to_target[edge_id]
        
        if math.isclose(t, end_time) and target_node is not None:
            polytomy_signature = (t, target_node)
            if polytomy_signature not in seen_polytomies:
                seen_polytomies.add(polytomy_signature)
                distinct_actions.append((edge_id, t))
        else:
            distinct_actions.append((edge_id, t))
            
    return distinct_actions

def enumerate_thread_choices(
    site_tree: SiteBackboneTree | Data,
    time_grid: Sequence[float],
) -> tuple[ThreadChoice, ...]:
    if isinstance(site_tree, Data):
        site = graph_site(site_tree)
        edge_index = site_tree.edge_index
        num_nodes = int(site_tree.num_nodes)
        root = graph_root(site_tree)
        node_times = site_tree.node_times
        branch_children = graph_branch_children(site_tree)
        branch_signatures = graph_branch_signatures(site_tree)
        parent_of_child = graph_parent_of_child(site_tree)
        descendant_signatures = graph_descendant_signatures(site_tree)
    else:
        site = site_tree.site
        edge_index = site_tree.edge_index
        num_nodes = site_tree.num_nodes
        root = site_tree.root
        node_times = site_tree.node_times
        branch_children = site_tree.branch_children
        branch_signatures = site_tree.branch_signatures
        parent_of_child = site_tree.parent_of_child
        descendant_signatures = site_tree.descendant_signatures

    backbone_edges = []
    for branch_child in branch_children:
        parent = parent_of_child[branch_child]
        backbone_edges.append({
            "edge_id": branch_child,
            "start_time": float(node_times[branch_child].item()),
            "end_time": float(node_times[parent].item()),
            "target_node_at_end": parent
        })
        
    backbone_edges.append({
        "edge_id": root,
        "start_time": float(node_times[root].item()),
        "end_time": float('inf'),
        "target_node_at_end": None
    })
    
    math_actions = get_mathematically_valid_actions(backbone_edges, list(time_grid))
    distinct_actions = get_distinct_topological_outcomes(math_actions, backbone_edges)
    distinct_set = set(distinct_actions)

    choices: list[ThreadChoice] = []
    time_eps = 1e-8
    for branch_child, branch_signature in zip(branch_children, branch_signatures):
        parent = parent_of_child[branch_child]
        child_time = float(node_times[branch_child].item())
        parent_time = float(node_times[parent].item())
        for time_idx, time_value in enumerate(time_grid):
            time_value_f = float(time_value)
            if (child_time + time_eps) < time_value_f < (parent_time - time_eps):
                if (branch_child, float(time_value)) in distinct_set:
                    choices.append(
                        ThreadChoice(
                            site=site,
                            branch_child=branch_child,
                            branch_signature=branch_signature,
                            time_idx=time_idx,
                            time_value=time_value_f,
                            is_root_branch=False,
                        )
                    )

    root_signature = descendant_signatures[root]
    root_time = float(node_times[root].item())
    for time_idx, time_value in enumerate(time_grid):
        if float(time_value) > root_time:
            if (root, float(time_value)) in distinct_set:
                choices.append(
                    ThreadChoice(
                        site=site,
                        branch_child=root,
                        branch_signature=root_signature,
                        time_idx=time_idx,
                        time_value=float(time_value),
                        is_root_branch=True,
                    )
                )
    return tuple(choices)


def _metadata_to_name(metadata: Any) -> Optional[str]:
    if metadata in (None, b"", ""):
        return None
    if isinstance(metadata, bytes):
        try:
            metadata = metadata.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return metadata if metadata else None
    if isinstance(metadata, dict):
        for key in ("name", "sample_id", "id"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _sample_leaf_names(ts: Any, sample_nodes: Sequence[int]) -> list[str]:
    leaf_names: list[str] = []
    for sample_node in sample_nodes:
        node = ts.node(sample_node)
        name = None
        if node.individual != -1:
            name = _metadata_to_name(ts.individual(node.individual).metadata)
        if name is None:
            name = _metadata_to_name(node.metadata)
        leaf_names.append(name if name is not None else str(sample_node))
    return leaf_names


def _timed_tree_from_tskit_tree(
    tree: Any,
    sample_id_by_node: Dict[int, int],
) -> TimedTree:
    roots = list(tree.roots)
    if len(roots) != 1:
        raise ValueError(
            f"tskit tree interval {tree.interval} has {len(roots)} roots; expected exactly 1"
        )

    def build(node_id: int) -> TimedTree:
        children = list(tree.children(node_id))
        sample_id = sample_id_by_node.get(node_id)
        if sample_id is not None:
            if children:
                raise ValueError(
                    f"Sample node {node_id} has children in interval {tree.interval}; "
                    "only leaf sample nodes are supported"
                )
            return sample_id
        if len(children) != 2:
            raise ValueError(
                f"Internal node {node_id} has {len(children)} children in interval "
                f"{tree.interval}; expected a binary local tree"
            )
        left, right = children
        return ("n", float(tree.time(node_id)), build(left), build(right))

    return build(roots[0])


def _graph_from_tskit_tree(
    tree: Any,
    sample_id_by_node: Dict[int, int],
    *,
    num_samples: int,
) -> Data:
    roots = list(tree.roots)
    if len(roots) != 1:
        raise ValueError(
            f"tskit tree interval {tree.interval} has {len(roots)} roots; expected exactly 1"
        )

    node_times: list[float] = []
    node_sample_ids: list[int] = []
    edges: list[tuple[int, int]] = []

    def allocate_node(time_value: float, sample_id: int) -> int:
        node_id = len(node_times)
        node_times.append(float(time_value))
        node_sample_ids.append(int(sample_id))
        return node_id

    def visit(node_id: int) -> int:
        children = list(tree.children(node_id))
        sample_id = sample_id_by_node.get(node_id)
        if sample_id is not None:
            if children:
                raise ValueError(
                    f"Sample node {node_id} has children in interval {tree.interval}; "
                    "only leaf sample nodes are supported"
                )
            return allocate_node(0.0, sample_id)
        if len(children) != 2:
            raise ValueError(
                f"Internal node {node_id} has {len(children)} children in interval "
                f"{tree.interval}; expected a binary local tree"
            )
        child_graph_ids = [visit(child_id) for child_id in children]
        graph_node_id = allocate_node(float(tree.time(node_id)), -1)
        for child_graph_id in child_graph_ids:
            edges.append((graph_node_id, child_graph_id))
        return graph_node_id

    root = visit(roots[0])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return make_tree_graph(
        edge_index=edge_index,
        num_nodes=len(node_times),
        root=root,
        node_times=torch.tensor(node_times, dtype=torch.float32),
        node_sample_ids=tuple(node_sample_ids),
        num_samples=num_samples,
    )


def _validate_binary_genotypes(geno: np.ndarray) -> None:
    if geno.ndim != 2:
        raise ValueError(f"Expected a 2D genotype matrix, got shape {geno.shape}")
    bad = np.setdiff1d(np.unique(geno), np.array([0, 1], dtype=geno.dtype))
    if bad.size:
        raise ValueError(
            "Only binary 0/1 tskit genotypes are supported; "
            f"found values {bad.tolist()}"
        )


def _validate_binary_variants(ts: Any) -> None:
    for variant in ts.variants():
        if len(variant.alleles) != 2 or any(allele is None for allele in variant.alleles):
            raise ValueError(
                "Only biallelic tskit variants without missing alleles are "
                f"supported; site {variant.site.id} has alleles {variant.alleles}"
            )


def _time_grid_from_tskit(ts: Any, time_grid_size: int) -> Tuple[float, ...]:
    if time_grid_size < 2:
        raise ValueError("time_grid_size must be at least 2")
    node_times = np.asarray(ts.tables.nodes.time, dtype=float)
    if node_times.size == 0:
        raise ValueError("The tree sequence has no nodes")
    return tuple(
        float(t)
        for t in np.linspace(
            float(node_times.min()),
            float(node_times.max()),
            time_grid_size,
        )
    )


def load_argweaver_sites_threading_inputs(
    sites_path: str,
    trees_path: str,
    time_grid_size: int,
) -> tuple[torch.Tensor, tuple[dict, ...], list[str], list[int], tuple[float, ...]]:
    """Load ARGweaver ``.sites`` observations with paired tskit local trees."""
    try:
        import tskit
    except ImportError as exc:
        raise ImportError("load_argweaver_sites_threading_inputs requires the tskit package") from exc

    sites = parse_argweaver_sites(sites_path)
    ts = tskit.load(trees_path)
    sample_nodes = list(ts.samples())
    if len(sample_nodes) != len(sites.leaf_names):
        raise ValueError(
            f"Paired .trees has {len(sample_nodes)} samples, but .sites has "
            f"{len(sites.leaf_names)} samples"
        )

    region_length = float(sites.region_end - sites.region_start + 1)
    if not math.isclose(float(ts.sequence_length), region_length):
        raise ValueError(
            "Paired .trees sequence_length does not match .sites REGION length: "
            f"{float(ts.sequence_length)} vs {region_length}"
        )

    time_grid = _time_grid_from_tskit(ts, time_grid_size)
    sample_id_by_node = {
        node_id: sample_id for sample_id, node_id in enumerate(sample_nodes)
    }
    site_positions = np.asarray(sites.site_positions, dtype=float)
    sequence_length = len(sites.site_positions)
    reference_full_trees: list[dict] = []

    for tree in ts.trees():
        left = float(tree.interval.left)
        right = float(tree.interval.right)
        start_site = int(np.searchsorted(site_positions, left, side="left"))
        end_site = int(np.searchsorted(site_positions, right, side="left"))
        if start_site == end_site:
            continue
        timed_tree = _timed_tree_from_tskit_tree(tree, sample_id_by_node)
        genomic_interval = (
            0.0 if start_site == 0 else float(site_positions[start_site]),
            region_length if end_site == sequence_length else float(site_positions[end_site]),
        )
        if reference_full_trees and reference_full_trees[-1]["tree"] == timed_tree:
            reference_full_trees[-1]["sites"] = (
                reference_full_trees[-1]["sites"][0],
                end_site,
            )
            reference_full_trees[-1]["genomic_interval"] = (
                reference_full_trees[-1]["genomic_interval"][0],
                genomic_interval[1],
            )
        else:
            reference_full_trees.append(
                {
                    "sites": (start_site, end_site),
                    "genomic_interval": genomic_interval,
                    "tree": timed_tree,
                }
            )

    if not reference_full_trees:
        raise ValueError("No local trees overlap ARGweaver variant sites")

    expected_start = 0
    for segment in reference_full_trees:
        start, end = segment["sites"]
        if start != expected_start:
            raise ValueError(
                "Converted paired .trees do not cover all ARGweaver variant-site indices; "
                f"expected segment to start at {expected_start}, got {start}"
            )
        expected_start = end
    if expected_start != sequence_length:
        raise ValueError(
            "Converted paired .trees do not cover all ARGweaver variant-site indices; "
            f"covered {expected_start} of {sequence_length}"
        )

    all_leaf_ids = list(range(len(sample_nodes)))
    return sites.geno, tuple(reference_full_trees), list(sites.leaf_names), all_leaf_ids, time_grid


def load_argweaver_sites_graph_inputs(
    sites_path: str,
    trees_path: str,
    time_grid_size: int,
) -> tuple[torch.Tensor, Tuple[GraphSegment, ...], list[str], list[int], tuple[float, ...]]:
    """Load ARGweaver ``.sites`` observations with paired tskit trees as PyG graphs."""
    try:
        import tskit
    except ImportError as exc:
        raise ImportError("load_argweaver_sites_graph_inputs requires the tskit package") from exc

    sites = parse_argweaver_sites(sites_path)
    ts = tskit.load(trees_path)
    sample_nodes = list(ts.samples())
    if len(sample_nodes) != len(sites.leaf_names):
        raise ValueError(
            f"Paired .trees has {len(sample_nodes)} samples, but .sites has "
            f"{len(sites.leaf_names)} samples"
        )

    region_length = float(sites.region_end - sites.region_start + 1)
    if not math.isclose(float(ts.sequence_length), region_length):
        raise ValueError(
            "Paired .trees sequence_length does not match .sites REGION length: "
            f"{float(ts.sequence_length)} vs {region_length}"
        )

    time_grid = _time_grid_from_tskit(ts, time_grid_size)
    sample_id_by_node = {
        node_id: sample_id for sample_id, node_id in enumerate(sample_nodes)
    }
    site_positions = np.asarray(sites.site_positions, dtype=float)
    sequence_length = len(sites.site_positions)
    reference_graph_segments: list[GraphSegment] = []

    for tree in ts.trees():
        left = float(tree.interval.left)
        right = float(tree.interval.right)
        start_site = int(np.searchsorted(site_positions, left, side="left"))
        end_site = int(np.searchsorted(site_positions, right, side="left"))
        if start_site == end_site:
            continue
        graph = _graph_from_tskit_tree(
            tree,
            sample_id_by_node,
            num_samples=len(sample_nodes),
        )
        genomic_interval = (
            0.0 if start_site == 0 else float(site_positions[start_site]),
            region_length if end_site == sequence_length else float(site_positions[end_site]),
        )
        if reference_graph_segments and graphs_equivalent(reference_graph_segments[-1].graph, graph):
            prev = reference_graph_segments[-1]
            reference_graph_segments[-1] = GraphSegment(
                start=prev.start,
                end=end_site,
                genomic_interval=(prev.genomic_interval or genomic_interval)[0:1] + (genomic_interval[1],),
                graph=prev.graph,
            )
        else:
            reference_graph_segments.append(
                GraphSegment(
                    start=start_site,
                    end=end_site,
                    genomic_interval=genomic_interval,
                    graph=graph,
                )
            )

    if not reference_graph_segments:
        raise ValueError("No local trees overlap ARGweaver variant sites")

    expected_start = 0
    for segment in reference_graph_segments:
        if segment.start != expected_start:
            raise ValueError(
                "Converted paired .trees do not cover all ARGweaver variant-site indices; "
                f"expected segment to start at {expected_start}, got {segment.start}"
            )
        expected_start = segment.end
    if expected_start != sequence_length:
        raise ValueError(
            "Converted paired .trees do not cover all ARGweaver variant-site indices; "
            f"covered {expected_start} of {sequence_length}"
        )

    all_leaf_ids = list(range(len(sample_nodes)))
    return sites.geno, tuple(reference_graph_segments), list(sites.leaf_names), all_leaf_ids, time_grid


def load_tskit_threading_inputs(
    trees_path: str,
    time_grid_size: int,
    fasta_path=None,
    full_sequence: bool = False,
) -> tuple[torch.Tensor, tuple[dict, ...], list[str], list[int], tuple[float, ...]]:
    """Load a tskit ``.trees`` file into the existing threading-env inputs.

    By default, the environment steps over tskit variant sites. With
    ``full_sequence=True``, the environment steps over every integer genomic
    position in the tree sequence and genotype features are encoded from FASTA.
    """
    if time_grid_size < 2:
        raise ValueError("time_grid_size must be at least 2")
    if full_sequence and fasta_path is None:
        raise ValueError("full_sequence=True requires fasta_path")

    try:
        import tskit
    except ImportError as exc:
        raise ImportError("load_tskit_threading_inputs requires the tskit package") from exc

    ts = tskit.load(trees_path)
    if not full_sequence and ts.num_sites == 0:
        raise ValueError("The tree sequence has no variant sites to use as env steps")

    sample_nodes = list(ts.samples())
    if not sample_nodes:
        raise ValueError("The tree sequence has no sample nodes")

    time_grid = _time_grid_from_tskit(ts, time_grid_size)

    sample_id_by_node = {
        node_id: sample_id for sample_id, node_id in enumerate(sample_nodes)
    }
    leaf_names = _sample_leaf_names(ts, sample_nodes)
    if full_sequence:
        sequence_length_float = float(ts.sequence_length)
        sequence_length = int(sequence_length_float)
        if sequence_length != sequence_length_float:
            raise ValueError(
                f"full_sequence=True requires integer ts.sequence_length, got {ts.sequence_length}"
            )
        fasta_sequences, fasta_leaf_names = _load_fasta_by_leaf_names(
            fasta_path,
            leaf_names=leaf_names,
            expected_count=len(sample_nodes),
            allow_order_fallback=True,
        )
        _validate_ordered_fasta_sequences(
            fasta_sequences,
            expected_count=len(sample_nodes),
            expected_length=sequence_length,
        )
        geno = _numeric_geno_from_fasta_sequences(fasta_sequences)
        leaf_names = list(fasta_leaf_names)
    else:
        geno_np = ts.genotype_matrix().T
        _validate_binary_genotypes(geno_np)
        _validate_binary_variants(ts)
        geno = torch.as_tensor(geno_np, dtype=torch.long)
        site_positions = np.asarray([site.position for site in ts.sites()], dtype=float)

    reference_full_trees: list[dict] = []

    for tree in ts.trees():
        if full_sequence:
            start_site = int(tree.interval.left)
            end_site = int(tree.interval.right)
            if start_site != float(tree.interval.left) or end_site != float(tree.interval.right):
                raise ValueError(
                    f"full_sequence=True requires integer tree intervals, got {tree.interval}"
                )
        else:
            left = float(tree.interval.left)
            right = float(tree.interval.right)
            start_site = int(np.searchsorted(site_positions, left, side="left"))
            end_site = int(np.searchsorted(site_positions, right, side="left"))
        if start_site == end_site:
            continue
        timed_tree = _timed_tree_from_tskit_tree(tree, sample_id_by_node)
        if reference_full_trees and reference_full_trees[-1]["tree"] == timed_tree:
            reference_full_trees[-1]["sites"] = (reference_full_trees[-1]["sites"][0], end_site)
        else:
            reference_full_trees.append({"sites": (start_site, end_site), "tree": timed_tree})

    if not reference_full_trees:
        raise ValueError("No local trees overlap variant sites")

    expected_start = 0
    for segment in reference_full_trees:
        start, end = segment["sites"]
        if start != expected_start:
            raise ValueError(
                "Converted tskit trees do not cover all variant-site indices; "
                f"expected segment to start at {expected_start}, got {start}"
            )
        expected_start = end
    expected_length = int(ts.sequence_length) if full_sequence else ts.num_sites
    if expected_start != expected_length:
        raise ValueError(
            "Converted tskit trees do not cover all environment site indices; "
            f"covered {expected_start} of {expected_length}"
        )

    all_leaf_ids = list(range(len(sample_nodes)))
    return geno, tuple(reference_full_trees), leaf_names, all_leaf_ids, time_grid
