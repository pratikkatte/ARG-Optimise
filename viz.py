"""Simple matplotlib visualization helpers for ARG states."""

from collections import defaultdict

import matplotlib.pyplot as plt


def draw_state(
    state,
    ax=None,
    output_path=None,
    show=True,
    title=None,
    show_segments=False,
    show_breakpoints=True,
):
    """Draw the graph represented by an ARGState.

    Nodes and edges are read from ``state.all_nodes``. Active lineages are
    highlighted with a red outline.
    """
    node_by_id = dict(state.all_nodes)
    active_ids = {lineage.node_id for lineage in state.active_lineages}
    depths = _compute_depths(node_by_id)
    positions = _compute_layered_positions(depths)

    if ax is None:
        max_layer_width = _max_layer_width(depths)
        max_depth = max(depths.values(), default=0)
        figsize = (max(6, 1.3 * max_layer_width + 2), max(4, 1.2 * (max_depth + 1) + 1))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_edges(ax, node_by_id, positions)
    _draw_nodes(ax, node_by_id, active_ids, positions, show_segments, show_breakpoints)
    _format_axes(ax, positions, title)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


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


def _draw_edges(ax, node_by_id, positions):
    for parent_id in sorted(node_by_id):
        parent_x, parent_y = positions[parent_id]
        for child_id in sorted(node_by_id[parent_id].children):
            if child_id not in positions:
                continue
            child_x, child_y = positions[child_id]
            ax.plot(
                [child_x, parent_x],
                [child_y, parent_y],
                color="0.45",
                linewidth=1.6,
                zorder=1,
            )


def _draw_nodes(ax, node_by_id, active_ids, positions, show_segments, show_breakpoints):
    for node_id in sorted(node_by_id):
        lineage = node_by_id[node_id]
        x, y = positions[node_id]
        is_leaf = len(lineage.children) == 0
        is_recomb_like = len(lineage.children) == 1
        is_active = node_id in active_ids

        marker = "s" if is_recomb_like else "o"
        facecolor = "forestgreen" if is_leaf else "white" if is_recomb_like else "royalblue"
        edgecolor = "red" if is_active else "royalblue" if is_recomb_like else "black"
        text_color = "black" if is_recomb_like else "white"

        ax.scatter(
            [x],
            [y],
            s=320,
            marker=marker,
            facecolors=facecolor,
            edgecolors=edgecolor,
            linewidths=2.6 if is_active else 1.4,
            zorder=3,
        )
        ax.text(
            x,
            y,
            str(node_id),
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
            zorder=4,
        )
        detail_label = _node_detail_label(lineage, show_segments, show_breakpoints)
        if detail_label:
            ax.text(
                x,
                y - 0.34,
                detail_label,
                ha="center",
                va="top",
                fontsize=7,
                color="0.15",
                zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
            )


def _node_detail_label(lineage, show_segments, show_breakpoints):
    details = []
    if show_breakpoints and lineage.event_type == "recomb" and lineage.breakpoint is not None:
        side = "" if lineage.recombination_side is None else f" {lineage.recombination_side[0].upper()}"
        details.append(f"b={lineage.breakpoint}{side}")
    if show_segments:
        details.append(_format_segments(_mask_to_segments(lineage.material_mask)))
    return "\n".join(detail for detail in details if detail)


def _format_segments(segments):
    if not segments:
        return "[]"
    return ",".join(f"[{start},{end})" for start, end in segments)


def _mask_to_segments(material_mask):
    segments = []
    start = None
    for block_i, has_material in enumerate(material_mask):
        if has_material and start is None:
            start = block_i
        elif not has_material and start is not None:
            segments.append((start, block_i))
            start = None
    if start is not None:
        segments.append((start, len(material_mask)))
    return segments


def _format_axes(ax, positions, title):
    if title is not None:
        ax.set_title(title)

    if positions:
        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)

    ax.set_xlabel("lineage order")
    ax.set_ylabel("event depth")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")
