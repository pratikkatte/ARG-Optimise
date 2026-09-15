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

    def is_valid_for(self, active_lineages):
        return 0 <= self.active_lineage_i < len(active_lineages) and self.span_start < self.span_end

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages):
        choices = []
        for i, lineage in enumerate(active_lineages):
            span = lineage.material_span
            if span is None:
                continue
            first_block, last_block, material_count = span
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

    def is_valid_for(self, active_lineages):
        i = self.active_lineage_i
        j = self.active_lineage_j
        return i != j and 0 <= i < len(active_lineages) and 0 <= j < len(active_lineages)

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages):
        return tuple(cls(i, j) for i in range(len(active_lineages))
                     for j in range(i + 1, len(active_lineages)))


@dataclass(frozen=True)
class PriorActionOptions:
    coal_actions: Tuple[CoalescenceChoice, ...]
    recomb_choices: Tuple[RecombinationChoice, ...]
    rates: Dict[str, float]

    @property
    def total_recomb_weight(self):
        return sum(choice.breakpoint_count
                   for choice in self.recomb_choices)
