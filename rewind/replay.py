"""Instability check: run the same computation repeatedly on identical inputs
and report whether the results are bitwise stable.

This is the confirmation step for suspected nondeterministic ops: a function
that produces varying digests from identical state is nondeterministic on this
hardware/config, full stop. RNG state is captured and restored around every
run by default, so ordinary RNG consumption does not read as instability.
"""

from dataclasses import dataclass, field

import torch

from . import hashing


@dataclass
class InstabilityReport:
    runs: int
    stable: bool
    outputs: int
    varying: list = field(default_factory=list)  # indices of unstable outputs
    digests: list = field(default_factory=list)  # per output: sorted unique hex digests

    def __str__(self):
        if self.stable:
            return f"stable over {self.runs} runs ({self.outputs} outputs)"
        return (
            f"UNSTABLE over {self.runs} runs: outputs {self.varying} vary "
            f"({len(self.varying)}/{self.outputs})"
        )


def _flatten(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        out = []
        for x in obj:
            out.extend(_flatten(x))
        return out
    if isinstance(obj, dict):
        out = []
        for k in sorted(obj, key=str):
            out.extend(_flatten(obj[k]))
        return out
    return []


def instability(fn, args=(), kwargs=None, runs: int = 20, restore_rng: bool = True):
    """Call fn(*args, **kwargs) `runs` times and compare output fingerprints."""
    kwargs = kwargs or {}
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    per_run = []
    for _ in range(runs):
        if restore_rng:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        out = fn(*args, **kwargs)
        per_run.append(
            [hashing.fingerprint_int(t) for t in _flatten(out)]
        )
    if restore_rng:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    n_out = max((len(r) for r in per_run), default=0)
    varying = []
    digests = []
    for i in range(n_out):
        seen = {r[i] for r in per_run if len(r) > i}
        digests.append(sorted(hashing.hex_digest(v) for v in seen))
        if len(seen) > 1 or any(len(r) <= i for r in per_run):
            varying.append(i)
    return InstabilityReport(runs, not varying, n_out, varying, digests)
