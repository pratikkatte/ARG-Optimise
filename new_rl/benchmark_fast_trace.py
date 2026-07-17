"""Local benchmark for the fast full-ARG trace builder."""

from __future__ import annotations

import argparse
import resource
import sys
import time

from arg.new_rl import build_fast_trace_from_full_arg


def _emit(message: str) -> None:
    print(message, flush=True)


def _rss_gib() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / 1024**3
    return rss / 1024**2


def _time_frontier(state, label: str) -> None:
    start = time.perf_counter()
    frontier = state.compact_active_frontier()
    elapsed = time.perf_counter() - start
    _emit(
        f"{label}_frontier_seconds\t{elapsed:.6f}"
        f"\tactive_lineages={len(frontier)}"
        f"\tsegments={frontier.segment_count}"
    )


def _move_with_progress(
    state,
    target_step: int,
    label: str,
    progress_events: int,
) -> float:
    target_step = int(target_step)
    progress_events = int(progress_events)
    start = time.perf_counter()
    if progress_events <= 0:
        state.move_to(target_step)
        return time.perf_counter() - start

    while state.step != target_step:
        if target_step > state.step:
            next_step = min(target_step, state.step + progress_events)
        else:
            next_step = max(target_step, state.step - progress_events)
        state.move_to(next_step)
        _emit(
            f"{label}_progress\tstep={state.step}"
            f"\telapsed={time.perf_counter() - start:.3f}"
        )
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees_path", help="Synthetic full-ARG .trees path")
    parser.add_argument(
        "--include-active",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--target-step",
        type=int,
        default=10_000,
        help="Intermediate cursor step to benchmark (default: 10000)",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Continue the cursor from the target step to the terminal state",
    )
    parser.add_argument(
        "--roundtrip",
        action="store_true",
        help="Backtrack the cursor to step zero after the forward benchmark",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=65_536,
        help="Number of events per Numba cursor chunk",
    )
    parser.add_argument(
        "--progress-events",
        type=int,
        default=1_000_000,
        help="Emit terminal/traceback progress after this many events; 0 disables it",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    trace = build_fast_trace_from_full_arg(args.trees_path)
    build_seconds = time.perf_counter() - start

    _emit(f"build_seconds\t{build_seconds:.3f}")
    _emit(f"build_max_rss_gib\t{_rss_gib():.3f}")
    _emit(f"events\t{trace.event_count}")
    _emit(f"recombination_events\t{trace.recombination_event_count}")
    _emit(f"coalescence_events\t{trace.coalescence_event_count}")
    _emit(f"nodes\t{trace.node_time.size}")
    _emit(f"edges\t{trace.edge_parent.size}")

    start = time.perf_counter()
    state = trace.initial_state(chunk_size=args.chunk_size)
    _emit(f"cursor_init_seconds\t{time.perf_counter() - start:.6f}")
    _time_frontier(state, "initial")

    # Compile both movement kernels before reporting transition throughput.
    if trace.num_steps:
        warmup = trace.initial_state(chunk_size=1)
        start = time.perf_counter()
        warmup.advance().backtrack()
        _emit(f"cursor_jit_warmup_seconds\t{time.perf_counter() - start:.6f}")

    target_step = min(max(int(args.target_step), 0), trace.num_steps)
    start = time.perf_counter()
    state.advance_to(target_step)
    _emit(
        f"cursor_advance_seconds\t{time.perf_counter() - start:.6f}"
        f"\tfrom=0\tto={target_step}"
    )
    _time_frontier(state, "target")

    if args.terminal and state.step < trace.num_steps:
        start_step = state.step
        elapsed = _move_with_progress(
            state,
            trace.num_steps,
            "cursor_terminal",
            args.progress_events,
        )
        _emit(
            f"cursor_terminal_seconds\t{elapsed:.6f}"
            f"\tfrom={start_step}\tto={trace.num_steps}"
        )
        _time_frontier(state, "terminal")

    if args.roundtrip:
        start_step = state.step
        elapsed = _move_with_progress(
            state,
            0,
            "cursor_backtrack",
            args.progress_events,
        )
        _emit(
            f"cursor_backtrack_seconds\t{elapsed:.6f}"
            f"\tfrom={start_step}\tto=0"
        )
        _time_frontier(state, "roundtrip")

    _emit(f"final_max_rss_gib\t{_rss_gib():.3f}")


if __name__ == "__main__":
    main()
