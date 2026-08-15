"""Diff two recorded runs: find the first divergent step, rank and module."""

from dataclasses import dataclass, field

from . import classify
from .ledger import Run

# Determinism-relevant header keys compared before any fingerprint work.
_CONFIG_KEYS = ("torch", "cuda", "cudnn", "devices", "world_size", "determinism", "env")


@dataclass
class Divergence:
    step: int
    rank: int
    kind: str  # batch | stream | module | mark | params | optim | rng
    module: str = ""
    phase: str = ""
    expected: str = ""
    observed: str = ""
    position: int = -1


@dataclass
class Report:
    identical: bool
    steps_compared: dict  # rank -> steps
    config_diff: dict  # key -> (a, b)
    first: Divergence | None
    cause: classify.Cause
    propagation: list = field(default_factory=list)
    infected: dict = field(default_factory=dict)  # rank -> first divergent step
    notes: list = field(default_factory=list)


def _config_diff(ha: dict, hb: dict) -> dict:
    out = {}
    for key in _CONFIG_KEYS:
        a, b = ha.get(key), hb.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    out[f"{key}.{k}"] = (a.get(k), b.get(k))
        elif a != b:
            out[key] = (a, b)
    return out


def _compare_step(a: dict, b: dict, rank: int) -> Divergence | None:
    """Compare two step records in execution order. Returns the first difference."""
    step = a.get("step", -1)

    if a.get("batch") != b.get("batch"):
        return Divergence(step, rank, "batch",
                          expected=str(a.get("batch")), observed=str(b.get("batch")))

    ma, mb = a.get("modules", []), b.get("modules", [])
    for i, (ea, eb) in enumerate(zip(ma, mb)):
        if ea[:2] != eb[:2]:
            return Divergence(step, rank, "stream", module=ea[0], phase=ea[1],
                              expected=f"{ea[0]}/{ea[1]}", observed=f"{eb[0]}/{eb[1]}",
                              position=i)
        if ea[2] != eb[2]:
            return Divergence(step, rank, "module", module=ea[0], phase=ea[1],
                              expected=ea[2], observed=eb[2], position=i)
    if len(ma) != len(mb):
        i = min(len(ma), len(mb))
        longer = ma if len(ma) > len(mb) else mb
        return Divergence(step, rank, "stream", module=longer[i][0], phase=longer[i][1],
                          expected=f"{len(ma)} entries", observed=f"{len(mb)} entries",
                          position=i)

    ka, kb = a.get("marks", {}), b.get("marks", {})
    for name in sorted(set(ka) | set(kb)):
        if ka.get(name) != kb.get(name):
            return Divergence(step, rank, "mark", module=name,
                              expected=str(ka.get(name)), observed=str(kb.get(name)))

    if a.get("params") != b.get("params"):
        da, db = a.get("params_detail"), b.get("params_detail")
        if da and db:
            for name in da:
                if da.get(name) != db.get(name):
                    return Divergence(step, rank, "params", module=name,
                                      expected=da[name], observed=db.get(name, ""))
        return Divergence(step, rank, "params",
                          expected=a.get("params", ""), observed=b.get("params", ""))

    if a.get("optim") != b.get("optim"):
        return Divergence(step, rank, "optim",
                          expected=a.get("optim", ""), observed=b.get("optim", ""))

    if a.get("rng") != b.get("rng"):
        return Divergence(step, rank, "rng",
                          expected=str(a.get("rng")), observed=str(b.get("rng")))
    return None


def _propagation(a: dict, b: dict, first: Divergence) -> list:
    """Names touched by the divergence within the step, in execution order."""
    chain = []
    if first.kind in ("module", "stream") and first.position >= 0:
        seen = set()
        pairs = list(zip(a.get("modules", []), b.get("modules", [])))
        for ea, eb in pairs[first.position :]:
            if ea[:2] == eb[:2] and ea[2] != eb[2] and ea[0] not in seen:
                chain.append(ea[0])
                seen.add(ea[0])
            if len(chain) >= 8:
                break
    ka, kb = a.get("marks", {}), b.get("marks", {})
    for name in sorted(set(ka) | set(kb)):
        if ka.get(name) != kb.get(name) and name not in chain:
            chain.append(name)
    if a.get("params") != b.get("params"):
        chain.append("params")
    if a.get("optim") != b.get("optim"):
        chain.append("optimizer")
    return chain


def diff_runs(dir_a: str, dir_b: str) -> Report:
    run_a, run_b = Run.load(dir_a), Run.load(dir_b)
    config_diff = _config_diff(run_a.header, run_b.header)

    notes = []
    firsts = {}
    steps_compared = {}
    common_ranks = sorted(set(run_a.ranks) & set(run_b.ranks))
    if not common_ranks:
        raise ValueError("runs share no ranks")
    only = set(run_a.ranks) ^ set(run_b.ranks)
    if only:
        notes.append(f"ranks not present in both runs, skipped: {sorted(only)}")

    for rank in common_ranks:
        ra, rb = run_a.ranks[rank], run_b.ranks[rank]
        n = min(len(ra), len(rb))
        steps_compared[rank] = n
        if len(ra) != len(rb):
            notes.append(
                f"rank {rank}: run lengths differ ({len(ra)} vs {len(rb)} steps), "
                f"compared the first {n}"
            )
        for a, b in zip(ra[:n], rb[:n]):
            d = _compare_step(a, b, rank)
            if d is not None:
                firsts[rank] = d
                break

    if not firsts:
        cause = classify.classify(None, config_diff, run_a.header, run_b.header)
        return Report(not config_diff, steps_compared, config_diff, None, cause,
                      notes=notes)

    origin_rank = min(firsts, key=lambda r: (firsts[r].step, r))
    first = firsts[origin_rank]
    infected = {
        r: d.step for r, d in firsts.items() if r != origin_rank
    }
    a_rec = run_a.ranks[origin_rank][first.step]
    b_rec = run_b.ranks[origin_rank][first.step]
    propagation = _propagation(a_rec, b_rec, first)
    cause = classify.classify(first, config_diff, run_a.header, run_b.header)
    return Report(False, steps_compared, config_diff, first, cause,
                  propagation, infected, notes)
