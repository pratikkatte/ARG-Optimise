import matplotlib.pyplot as plt
import torch

from typing import Tuple, Optional, Sequence, Dict, Any

def _read_attr(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def draw_tree_edge_index(state, leaf_names=None, ax=None, title: str = ""):
    edge_index = _read_attr(state, "edge_index")
    num_nodes = _read_attr(state, "num_nodes")
    root = _read_attr(state, "root")
    node_times = _read_attr(state, "node_times")
    node_sample_ids = _read_attr(state, "node_sample_ids")
    highlight_edges = set(tuple(edge) for edge in (_read_attr(state, "highlight_edges", ()) or ()))
    highlight_samples = set(_read_attr(state, "highlight_samples", ()) or ())

    if edge_index is None or num_nodes is None or root is None:
        raise ValueError("draw_tree_edge_index requires edge_index, num_nodes, and root")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.set_title(title)

    if edge_index.numel() == 0:
        label = str(root)
        ax.text(0, 0, label, ha="center", va="center", fontsize=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return ax

    children = [[] for _ in range(num_nodes)]
    for parent, child in edge_index.t().tolist():
        children[parent].append(child)

    order = []

    def collect(node: int):
        if not children[node]:
            order.append(node)
            return
        for child in children[node]:
            collect(child)

    collect(root)
    x_pos = {leaf: idx for idx, leaf in enumerate(order)}

    def label_leaf(node_id: int) -> str:
        if node_sample_ids is not None:
            sample_id = node_sample_ids[node_id]
            if sample_id >= 0:
                if leaf_names and sample_id < len(leaf_names):
                    return leaf_names[sample_id]
                return str(sample_id)
        if leaf_names and node_id < len(leaf_names):
            return leaf_names[node_id]
        return str(node_id)

    def draw(node: int, y: float) -> Tuple[float, float]:
        if not children[node]:
            x = x_pos[node]
            text_color = "darkorange" if node_sample_ids is not None and node_sample_ids[node] in highlight_samples else "black"
            ax.text(x, y, label_leaf(node), ha="center", va="center", fontsize=10, color=text_color)
            return x, y

        coords = []
        for child in children[node]:
            coords.append((child, *draw(child, y - 1)))

        x = sum(child_x for _, child_x, _ in coords) / len(coords)
        for child, child_x, child_y in coords:
            edge_color = "darkorange" if (node, child) in highlight_edges else "steelblue"
            edge_width = 2.5 if (node, child) in highlight_edges else 1.0
            ax.plot([child_x, child_x], [child_y, y - 0.2], linewidth=edge_width, color=edge_color)
        ax.plot(
            [coords[0][1], coords[-1][1]],
            [y - 0.2, y - 0.2],
            linewidth=1.0,
            color="steelblue",
        )
        ax.plot([x, x], [y - 0.2, y], linewidth=1.0, color="steelblue")

        if node_times is not None:
            label = f"{node}\nt={float(node_times[node]):.2f}"
        else:
            label = str(node)
        ax.text(x, y - 0.05, label, ha="center", va="bottom", fontsize=8, color="firebrick")
        return x, y

    draw(root, y=0)
    ax.set_xlim(-1, max(len(order), 1))
    return ax


def draw_local_tree_sequence(tree_sequence, leaf_names=None, title: str = ""):
    if not tree_sequence:
        return None

    fig, axes = plt.subplots(1, len(tree_sequence), figsize=(6 * len(tree_sequence), 3))
    if len(tree_sequence) == 1:
        axes = [axes]

    if title:
        fig.suptitle(title)
    for ax, tree in zip(axes, tree_sequence):
        print(tree)
        start = _read_attr(tree, "start")
        end = _read_attr(tree, "end")
        choice = _read_attr(tree, "choice")
        interval = f"sites [{start}:{end})" if start is not None and end is not None else ""
        if choice is not None:
            branch_label = "root" if getattr(choice, "is_root_branch", False) else getattr(choice, "branch_signature", "")
            interval = f"{interval}\nbranch {branch_label} @ t{choice.time_idx}"
     
        draw_tree_edge_index(tree, leaf_names=leaf_names, ax=ax, title=interval)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    plt.show()
    # plt.close(fig)
    # return fig
