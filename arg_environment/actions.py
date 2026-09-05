import numbers
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass(frozen=True)
class RecombinationChoice:
    active_lineage_i: int
    material_count: int
    span_start: int
    span_end: int
    time_action: Optional[int] = None
    breakpoint: Optional[int] = None

    @property
    def breakpoint_count(self):
        return int(self.span_end - self.span_start)

    def as_weight_tuple(self):
        return (
            self.active_lineage_i,
            self.material_count,
            list(range(self.span_start + 1, self.span_end + 1)),
        )

    @classmethod
    def from_action(cls, action):
        if isinstance(action, cls):
            return action
        if not isinstance(action, dict) or action.get("event_type") != "recomb":
            return None
        active_lineage_i = action.get("active_lineage_i")
        material_count = action.get("material_count")
        span_start = action.get("span_start")
        span_end = action.get("span_end")
        if not isinstance(active_lineage_i, numbers.Integral) or not isinstance(material_count, numbers.Integral) or not isinstance(span_start, numbers.Integral) or not isinstance(span_end, numbers.Integral):
            return None
        time_action = action.get("time_action")
        if time_action is not None and not isinstance(time_action, numbers.Integral):
            return None
        breakpoint = action.get("breakpoint")
        if breakpoint is not None and not isinstance(breakpoint, numbers.Integral):
            return None
        return cls(
            active_lineage_i=int(active_lineage_i),
            material_count=int(material_count),
            span_start=int(span_start),
            span_end=int(span_end),
            time_action=int(time_action) if time_action is not None else None,
            breakpoint=int(breakpoint) if breakpoint is not None else None,
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
    active_lineage_i: int
    active_lineage_j: int
    time_action: Optional[int] = None

    def as_dict(self):
        action = {
            "event_type": "coal",
            "active_lineage_i": self.active_lineage_i,
            "active_lineage_j": self.active_lineage_j,
        }
        if self.time_action is not None:
            action["time_action"] = self.time_action
        return action

    def is_valid_for(self, active_lineages):
        i = self.active_lineage_i
        j = self.active_lineage_j
        if i == j:
            return False
        if not (0 <= i < len(active_lineages) and 0 <= j < len(active_lineages)):
            return False
        return active_lineages[i].material_segments.overlaps(
            active_lineages[j].material_segments
        )

    @classmethod
    def from_action(cls, action):
        if isinstance(action, cls):
            return action
        if not isinstance(action, dict) or action.get("event_type") != "coal":
            return None
        i = action.get("active_lineage_i")
        j = action.get("active_lineage_j")
        if not isinstance(i, numbers.Integral) or not isinstance(j, numbers.Integral):
            return None
        time_action = action.get("time_action")
        if time_action is not None and not isinstance(time_action, numbers.Integral):
            return None
        return cls(
            active_lineage_i=int(i),
            active_lineage_j=int(j),
            time_action=int(time_action) if time_action is not None else None,
        )

    @classmethod
    def enumerate_from_active_lineages(cls, active_lineages):
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

    @property
    def total_recomb_weight(self):
        return sum(choice.material_count for choice in self.recomb_choices)

