"""Order-sensitive 64-bit fingerprints of tensor bit patterns.

Digests are returned as 0-dim int64 tensors on the input's device so callers
can batch the device-to-host transfer. Plain XOR folding is permutation
invariant and cancels paired bit flips, so every word is salted with its
position before the fold.
"""

import torch

_M64 = (1 << 64) - 1
_P1U = 0x9E3779B97F4A7C15
_P2U = 0xC2B2AE3D27D4EB4F
_P1 = _P1U - (1 << 64)
_P2 = _P2U - (1 << 64)
_CHUNK = 1 << 22  # 64-bit words per chunk; bounds temp memory to 32 MiB


def _mix(x: torch.Tensor) -> torch.Tensor:
    x = x ^ ((x >> 33) & 0x7FFFFFFF)
    x = x * _P1
    x = x ^ ((x >> 29) & 0x7FFFFFFFF)
    x = x * _P2
    x = x ^ ((x >> 32) & 0xFFFFFFFF)
    return x


def _mix_int(x: int) -> int:
    x &= _M64
    x ^= x >> 33
    x = (x * _P1U) & _M64
    x ^= x >> 29
    x = (x * _P2U) & _M64
    x ^= x >> 32
    return x


def _hash_str(s: str) -> int:
    h = len(s)
    for b in s.encode():
        h = _mix_int(h ^ b)
    return h


def _to_signed(x: int) -> int:
    x &= _M64
    return x - (1 << 64) if x >= (1 << 63) else x


def _xor_fold(v: torch.Tensor) -> torch.Tensor:
    n = v.numel()
    while n > 1:
        half = (n + 1) // 2
        a = v[:half].clone()
        b = v[half:]
        a[: b.numel()] ^= b
        v = a
        n = half
    return v.reshape(())


def fingerprint(t: torch.Tensor) -> torch.Tensor:
    """Bitwise fingerprint of a tensor. Returns a 0-dim int64 tensor on t's device."""
    with torch.no_grad():
        t = t.detach()
        if t.is_sparse:
            t = t.to_dense()
        meta = _hash_str(str(t.dtype))
        for d in t.shape:
            meta = _mix_int(meta ^ d)
        if t.dtype == torch.bool:
            t = t.to(torch.uint8)
        t = t.contiguous().flatten()
        dev = t.device
        h = torch.tensor(_to_signed(meta), dtype=torch.int64, device=dev)
        if t.numel() == 0:
            return _mix(h)
        raw = t.view(torch.uint8)
        if raw.storage_offset() % 8:
            # int64 view below needs 8-byte-aligned storage; offset views
            # (x[1:], chunk, narrow) are contiguous but not aligned
            raw = raw.clone()
        pad = (-raw.numel()) % 8
        if pad:
            raw = torch.cat([raw, raw.new_zeros(pad)])
        words = raw.view(torch.int64)
        for start in range(0, words.numel(), _CHUNK):
            chunk = words[start : start + _CHUNK]
            idx = torch.arange(
                start, start + chunk.numel(), dtype=torch.int64, device=dev
            )
            h = _mix(h ^ _xor_fold(_mix(chunk ^ _mix(idx))))
        return h


def fingerprint_int(t: torch.Tensor) -> int:
    """fingerprint(), synced to a python int. One host sync per call."""
    return int(fingerprint(t).cpu())


def combine(digests: list) -> torch.Tensor:
    """Order-sensitive fold of digest tensors into one. Empty list hashes to a constant."""
    if not digests:
        return torch.tensor(_to_signed(_mix_int(0)), dtype=torch.int64)
    h = digests[0]
    for d in digests[1:]:
        h = _mix(h ^ d.to(h.device))
    return _mix(h)


def combine_ints(values: list) -> int:
    h = len(values)
    for v in values:
        h = _mix_int(h ^ (v & _M64))
    return h


def hex_digest(v: int) -> str:
    return f"{v & _M64:016x}"
