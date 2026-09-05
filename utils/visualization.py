"""Simple matplotlib visualization helpers for ARG states."""

import matplotlib.pyplot as plt

from .visual_layout import (
    _all_visual_positions, _build_visual_model, _compute_depths,
    _compute_layered_positions, _max_layer_width,
)
from .visual_render import _draw_edges, _draw_legend, _draw_nodes, _draw_recombination_events, _format_axes


def draw_state(
    state,
    ax=None,
    output_path=None,
    show=True,
    title=None,
    show_segments=False,
    show_breakpoints=True,
    show_legend=True,
):
    """Draw the graph represented by an ARGState.

    Nodes and edges are read from ``state.all_nodes``. Active lineages are
    highlighted with a red outline.
    """
    node_by_id = dict(state.all_nodes)
    active_ids = {lineage.node_id for lineage in state.active_lineages}
    depths = _compute_depths(node_by_id)
    positions = _compute_layered_positions(depths)
    visual_model = _build_visual_model(node_by_id, positions)

    if ax is None:
        max_layer_width = _max_layer_width(depths)
        max_depth = max(depths.values(), default=0)
        figsize = (max(6, 1.3 * max_layer_width + 2), max(4, 1.2 * (max_depth + 1) + 1))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _draw_edges(ax, node_by_id, positions, visual_model)
    _draw_nodes(ax, node_by_id, active_ids, positions, visual_model, show_segments, show_breakpoints)
    _draw_recombination_events(ax, visual_model, show_breakpoints)
    if show_legend:
        _draw_legend(ax)
    _format_axes(ax, _all_visual_positions(positions, visual_model), title)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

