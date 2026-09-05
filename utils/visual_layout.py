from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class _RecombinationEvent:
    child_id: int
    breakpoint: int
    parent_ids: tuple


@dataclass
class _VisualModel:
    recombination_events: list
    recombination_positions: dict
    recombination_parent_ids: set
    grouped_recombination_edges: set
def _compute_depths(node_by_id):
    depths = {}
    visiting = set()

    def depth(node_id):
        if node_id in depths:
            return depths[node_id]
        if node_id in visiting:
            raise ValueError("ARG graph contains a cycle")

        visiting.add(node_id)
        children = [
            child_id
            for child_id in node_by_id[node_id].children
            if child_id in node_by_id
        ]
        if not children:
            node_depth = 0
        else:
            node_depth = 1 + max(depth(child_id) for child_id in children)
        visiting.remove(node_id)
        depths[node_id] = node_depth
        return node_depth

    for node_id in sorted(node_by_id):
        depth(node_id)
    return depths


def _compute_layered_positions(depths):
    layers = defaultdict(list)
    for node_id, depth in depths.items():
        layers[depth].append(node_id)

    positions = {}
    for depth, node_ids in layers.items():
        sorted_node_ids = sorted(node_ids)
        layer_width = len(sorted_node_ids)
        for idx, node_id in enumerate(sorted_node_ids):
            x = idx - (layer_width - 1) / 2.0
            positions[node_id] = (x, float(depth))
    return positions


def _max_layer_width(depths):
    layer_counts = defaultdict(int)
    for depth in depths.values():
        layer_counts[depth] += 1
    return max(layer_counts.values(), default=1)


def _build_visual_model(node_by_id, positions):
    recombination_events = _group_recombination_events(node_by_id)
    recombination_positions = _compute_recombination_event_positions(recombination_events, positions)
    recombination_parent_ids = {
        parent_id
        for event in recombination_events
        for parent_id in event.parent_ids
    }
    grouped_recombination_edges = {
        (parent_id, event.child_id)
        for event in recombination_events
        for parent_id in event.parent_ids
    }
    return _VisualModel(
        recombination_events=recombination_events,
        recombination_positions=recombination_positions,
        recombination_parent_ids=recombination_parent_ids,
        grouped_recombination_edges=grouped_recombination_edges,
    )


def _group_recombination_events(node_by_id):
    grouped_parent_ids = defaultdict(list)
    for node_id, lineage in sorted(node_by_id.items()):
        if lineage.event_type != "recomb" or not lineage.children:
            continue
        child_id = lineage.children[0]
        if child_id not in node_by_id:
            continue
        grouped_parent_ids[(child_id, lineage.breakpoint)].append(node_id)

    events = []
    for (child_id, breakpoint), parent_ids in sorted(
        grouped_parent_ids.items(),
        key=lambda item: (_optional_int_sort_key(item[0][0]), _optional_int_sort_key(item[0][1])),
    ):
        if len(parent_ids) < 2:
            continue
        events.append(
            _RecombinationEvent(
                child_id=child_id,
                breakpoint=breakpoint,
                parent_ids=tuple(sorted(parent_ids)),
            )
        )
    return events


def _optional_int_sort_key(value):
    return -1 if value is None else value


def _compute_recombination_event_positions(recombination_events, positions):
    recombination_positions = {}
    for event in recombination_events:
        if event.child_id not in positions:
            continue
        child_x, child_y = positions[event.child_id]
        parent_ys = [
            positions[parent_id][1]
            for parent_id in event.parent_ids
            if parent_id in positions
        ]
        if parent_ys:
            closest_parent_y = min(parent_ys)
            event_y = child_y + max(0.35, 0.45 * (closest_parent_y - child_y))
            if event_y >= closest_parent_y:
                event_y = (child_y + closest_parent_y) / 2.0
        else:
            event_y = child_y + 0.45
        recombination_positions[event] = (child_x, event_y)
    return recombination_positions


def _all_visual_positions(positions, visual_model):
    all_positions = dict(positions)
    for idx, event in enumerate(visual_model.recombination_events):
        if event in visual_model.recombination_positions:
            all_positions[f"recomb_{idx}"] = visual_model.recombination_positions[event]
    return all_positions



