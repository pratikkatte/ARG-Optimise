"""State-local physical choices and immutable incremental compatibility snapshots."""
from dataclasses import dataclass

import numpy as np

from .actions import CoalescenceChoice, RecombinationChoice


@dataclass(frozen=True)
class ActionContext:
    signature: tuple
    coal_actions: tuple
    recomb_choices: tuple
    recomb_by_lineage: tuple
    rates: dict

    @classmethod
    def build(cls, state, signature, compute_rates):
        coal, recomb = ((), ()) if state.is_done else (
            CoalescenceChoice.enumerate_from_active_lineages(state.active_lineages),
            RecombinationChoice.enumerate_from_active_lineages(state.active_lineages))
        by_lineage = [None] * len(state.active_lineages)
        for choice in recomb:
            by_lineage[choice.active_lineage_i] = choice
        return cls(signature, coal, recomb, tuple(by_lineage), compute_rates((coal, recomb)))


@dataclass(frozen=True, eq=False)
class CompatibilityCache:
    dataset_fingerprint: str
    lineages: tuple
    descendants: np.ndarray
    support: np.ndarray
    coal_actions: tuple


def update_compatibility(env, context, previous):
    """Only evaluate pairs involving new or changed lineages.

    Signatures retain immutable material/descendant values, not mutable lineage
    objects. Snapshots can therefore be shared by cloned states without retaining
    previous states or changing a sibling branch's support.
    """
    lineages = context.signature[-1]
    dtype = np.uint64 if env.num_sequences <= 64 else object
    descendants = np.zeros((len(lineages), env.num_variants), dtype=dtype)
    support = np.ones((len(lineages), len(lineages)), dtype=bool)
    if previous is not None and previous.dataset_fingerprint != env.dataset_fingerprint:
        previous = None
    old_rows = {} if previous is None else {entry[0]: i for i, entry in enumerate(previous.lineages)}
    kept, kept_old, changed_rows = [], [], set()
    for row, entry in enumerate(lineages):
        old = old_rows.get(entry[0])
        if old is not None and previous.lineages[old] == entry:
            kept.append(row)
            kept_old.append(old)
        else:
            changed_rows.add(row)
    if kept:
        descendants[kept] = previous.descendants[kept_old]
        support[np.ix_(kept, kept)] = previous.support[np.ix_(kept_old, kept_old)]
    for row in changed_rows:
        for left, right, bits in lineages[row][2].segments:
            start, end = np.searchsorted(env.snp_data.positions, [left, right])
            descendants[row, start:end] = bits
    targets = np.asarray(env.derived_sets, dtype=dtype)
    changed_pairs = [a for a in context.coal_actions
                     if a.active_lineage_i in changed_rows or a.active_lineage_j in changed_rows]
    for start in range(0, len(changed_pairs), 256):
        choices = changed_pairs[start:start+256]
        ii = [a.active_lineage_i for a in choices]
        jj = [a.active_lineage_j for a in choices]
        left, right = descendants[ii], descendants[jj]
        union = left | right
        conflicts = ((left != 0) & (right != 0) & ((union & targets) != 0)
                     & ((union & ~targets) != 0) & ((targets & ~union) != 0)).any(axis=1)
        support[ii, jj] = ~conflicts
        support[jj, ii] = ~conflicts
    allowed = tuple(a for a in context.coal_actions if support[a.active_lineage_i, a.active_lineage_j])
    for array in (descendants, support):
        array.setflags(write=False)
    return CompatibilityCache(env.dataset_fingerprint, lineages, descendants, support, allowed)
