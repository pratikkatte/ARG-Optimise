from matplotlib.lines import Line2D


def _draw_edges(ax, node_by_id, positions, visual_model):
    for parent_id in sorted(node_by_id):
        parent_x, parent_y = positions[parent_id]
        for child_id in sorted(node_by_id[parent_id].children):
            if child_id not in positions:
                continue
            if (parent_id, child_id) in visual_model.grouped_recombination_edges:
                continue
            child_x, child_y = positions[child_id]
            _draw_edge(ax, child_x, child_y, parent_x, parent_y)

    for event in visual_model.recombination_events:
        if event not in visual_model.recombination_positions or event.child_id not in positions:
            continue
        event_x, event_y = visual_model.recombination_positions[event]
        child_x, child_y = positions[event.child_id]
        _draw_edge(ax, child_x, child_y, event_x, event_y)
        for parent_id in event.parent_ids:
            if parent_id not in positions:
                continue
            parent_x, parent_y = positions[parent_id]
            _draw_edge(ax, event_x, event_y, parent_x, parent_y)


def _draw_edge(ax, child_x, child_y, parent_x, parent_y):
    ax.plot(
        [child_x, parent_x],
        [child_y, parent_y],
        color="0.45",
        linewidth=1.6,
        zorder=1,
    )


def _draw_nodes(ax, node_by_id, active_ids, positions, visual_model, show_segments, show_breakpoints):
    for node_id in sorted(node_by_id):
        lineage = node_by_id[node_id]
        x, y = positions[node_id]
        is_active = node_id in active_ids
        node_kind = _node_visual_kind(lineage, node_id, visual_model)
        if node_kind == "coalescence":
            _draw_coalescence_node(ax, x, y, node_id, is_active)
        elif node_kind == "recombination":
            _draw_recombination_node(ax, x, y, node_id, lineage, is_active, show_breakpoints)
        else:
            _draw_lineage_endpoint(ax, x, y, node_id, lineage, is_active, show_segments)


def _node_visual_kind(lineage, node_id, visual_model):
    if node_id in visual_model.recombination_parent_ids:
        return "lineage"
    if lineage.event_type == "coal":
        return "coalescence"
    if lineage.event_type == "recomb":
        return "recombination"
    return "lineage"


def _draw_coalescence_node(ax, x, y, node_id, is_active):
    ax.scatter(
        [x],
        [y],
        s=340,
        marker="o",
        facecolors="royalblue",
        edgecolors="red" if is_active else "black",
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
        color="white",
        zorder=4,
    )


def _draw_recombination_node(ax, x, y, node_id, lineage, is_active, show_breakpoints):
    _draw_recombination_square(
        ax,
        x,
        y,
        _recombination_label(lineage.breakpoint, show_breakpoints),
        edgecolor="red" if is_active else "darkorange",
        linewidth=2.6 if is_active else 1.6,
    )
    _draw_offset_label(ax, x, y, str(node_id), y_offset=-0.32)


def _draw_lineage_endpoint(ax, x, y, node_id, lineage, is_active, show_segments):
    ax.scatter(
        [x],
        [y],
        s=120,
        marker="o",
        facecolors="white",
        edgecolors="red" if is_active else "0.35",
        linewidths=2.4 if is_active else 1.2,
        zorder=3,
    )
    _draw_offset_label(ax, x, y, str(node_id), x_offset=0.13, y_offset=0.02)
    detail_label = _node_detail_label(lineage, show_segments)
    if detail_label:
        _draw_offset_label(
            ax,
            x,
            y,
            detail_label,
            x_offset=0.13,
            y_offset=-0.18,
            fontsize=7,
            va="top",
        )


def _draw_recombination_events(ax, visual_model, show_breakpoints):
    for event in visual_model.recombination_events:
        if event not in visual_model.recombination_positions:
            continue
        x, y = visual_model.recombination_positions[event]
        _draw_recombination_square(
            ax,
            x,
            y,
            _recombination_label(event.breakpoint, show_breakpoints),
            edgecolor="darkorange",
            linewidth=1.8,
        )


def _draw_recombination_square(ax, x, y, label, edgecolor, linewidth):
    ax.scatter(
        [x],
        [y],
        s=430,
        marker="s",
        facecolors="white",
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=4,
    )
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=7 if "\n" in label else 9,
        color="0.1",
        linespacing=0.9,
        zorder=5,
    )


def _recombination_label(breakpoint, show_breakpoints):
    if not show_breakpoints or breakpoint is None:
        return "R"
    return f"R\nb={breakpoint}"


def _draw_offset_label(ax, x, y, label, x_offset=0.0, y_offset=0.0, fontsize=8, va="center"):
    ax.text(
        x + x_offset,
        y + y_offset,
        label,
        ha="left" if x_offset else "center",
        va=va,
        fontsize=fontsize,
        color="0.15",
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
    )


def _node_detail_label(lineage, show_segments):
    details = []
    if show_segments:
        segment_label = _format_segments(_mask_to_segments(lineage.material_mask))
        if lineage.event_type == "recomb" and lineage.recombination_side:
            segment_label = f"{lineage.recombination_side[0].upper()}: {segment_label}"
        details.append(segment_label)
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


def _draw_legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="royalblue",
            markeredgecolor="black",
            markersize=8,
            label="Coalescence",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor="darkorange",
            markersize=8,
            label="Recombination",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="0.35",
            markersize=5,
            label="Sample/lineage",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="red",
            markeredgewidth=2,
            markersize=6,
            label="Active lineage",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        borderpad=0.2,
    )


