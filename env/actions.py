from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple


@dataclass(frozen=True)
class RecombinationChoice:
    event_type: ClassVar[str] = "recomb"
    active_lineage_i: int
    material_count: int
    span_start: int
    span_end: int
    time_action: Optional[int] = None
    breakpoint: Optional[int] = None
    delta_t: Optional[float] = None

    @property
    def breakpoint_count(self):
        return int(self.span_end - self.span_start)

    def as_weight_tuple(self):
        return (
            self.active_lineage_i,
            self.material_count,
            list(range(self.span_start + 1, self.span_end + 1)),
        )

    def is_valid_for(self, active_lineages):
        return self.active_lineage_i < len(active_lineages) and self.span_start < self.span_end

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages):
        choices = []
        for i, lineage in enumerate(active_lineages):
            span = lineage.material_span
            if span is None:
                continue
            first_block, last_block, material_count = span
            if first_block < last_block:
                choices.append(
                    cls(
                        active_lineage_i=i,
                        material_count=int(material_count),
                        span_start=int(first_block),
                        span_end=int(last_block),
                    )
                )
        return tuple(choices)


@dataclass(frozen=True)
class CoalescenceChoice:
    event_type: ClassVar[str] = "coal"
    active_lineage_i: int
    active_lineage_j: int
    time_action: Optional[int] = None
    delta_t: Optional[float] = None

    def is_valid_for(self, active_lineages, allow_nonoverlap=False):
        i = self.active_lineage_i
        j = self.active_lineage_j
        if i == j:
            return False
        if not (0 <= i < len(active_lineages) and 0 <= j < len(active_lineages)):
            return False
        return allow_nonoverlap or active_lineages[i].material_segments.overlaps(
            active_lineages[j].material_segments
        )

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages, allow_nonoverlap=False):
        if allow_nonoverlap:
            return tuple(cls(i, j) for i in range(len(active_lineages))
                         for j in range(i + 1, len(active_lineages)))
        events = []
        for active_idx, lineage in enumerate(active_lineages):
            for start, end in lineage.material_segments.segments:
                events.append((start, 1, active_idx))
                events.append((end, -1, active_idx))
        events.sort(key=lambda item: (item[0], item[1]))

        active = set()
        pairs = set()
        for _position, event_type, active_idx in events:
            if event_type < 0:
                active.discard(active_idx)
                continue
            for other_idx in active:
                if other_idx < active_idx:
                    pairs.add((other_idx, active_idx))
                else:
                    pairs.add((active_idx, other_idx))
            active.add(active_idx)

        return tuple(
            cls(active_lineage_i=i, active_lineage_j=j)
            for i, j in sorted(pairs)
        )


@dataclass(frozen=True)
class PriorActionOptions:
    coal_actions: Tuple[CoalescenceChoice, ...]
    recomb_choices: Tuple[RecombinationChoice, ...]
    rates: Dict[str, float]
    arg_prior: str = 'overlap'

    @property
    def total_recomb_weight(self):
        return sum(choice.breakpoint_count if self.arg_prior == 'hudson' else choice.material_count
                   for choice in self.recomb_choices)
