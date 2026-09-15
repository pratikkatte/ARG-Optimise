"""Bounded reuse of unchanged, non-differentiable lineage input features.

Only material masking and row normalization are cached. Every learned encoder
and action/flow head is evaluated afresh. Weak source references release cached
rows when ancestral partials disappear, and tensor versions guard in-place edits.
"""
from collections import OrderedDict
import weakref

import torch


class LineageFeatureCache:
    def __init__(self, max_bytes=128 * 1024 * 1024):
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.bypasses = 0

    def _remove(self, key, ref=None):
        entry = self.entries.get(key)
        if entry is not None and (ref is None or entry[0] is ref):
            self.entries.pop(key)
            self.bytes -= entry[2]

    def get(self, lineage, device, num_blocks, compute):
        source = lineage.partials
        if not torch.is_tensor(source) or source.requires_grad:
            self.bypasses += 1
            return compute()
        try:
            version = source._version
        except RuntimeError:
            # Inference tensors do not track mutations: do not cache them.
            self.bypasses += 1
            return compute()
        key = id(source)
        signature = (version, lineage.material_segments.segments, torch.device(device),
                     int(num_blocks), source.dtype, tuple(source.shape), source.data_ptr())
        entry = self.entries.get(key)
        if entry is not None and entry[0]() is source and entry[1] == signature:
            self.entries.move_to_end(key)
            self.hits += 1
            return entry[3]
        value = compute()
        self.misses += 1
        self._remove(key)
        size = value.numel() * value.element_size()
        if value.requires_grad or size > self.max_bytes:
            return value
        while self.entries and self.bytes + size > self.max_bytes:
            self._remove(next(iter(self.entries)))
        cache_ref = weakref.ref(self)

        def release(ref):
            cache = cache_ref()
            if cache is not None:
                cache._remove(key, ref)

        self.entries[key] = (weakref.ref(source, release), signature, size, value)
        self.bytes += size
        return value

    def clear(self):
        self.entries.clear()
        self.bytes = 0
