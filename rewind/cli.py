"""rewind CLI: diff and show."""

import argparse
import sys

from .diff import diff_runs
from .ledger import Run


def _print_report(rep) -> None:
    if rep.config_diff:
        print("config diff")
        for k, (a, b) in sorted(rep.config_diff.items()):
            print(f"  {k}")
            print(f"    a  {a}")
            print(f"    b  {b}")
        print()

    if rep.first is None:
        n = ", ".join(f"rank {r}: {n}" for r, n in sorted(rep.steps_compared.items()))
        if rep.identical:
            print(f"runs identical ({n} steps)")
        else:
            print(f"fingerprints identical ({n} steps) but configs differ")
            print()
            print(f"cause  {rep.cause.name}")
            print(f"       {rep.cause.detail}")
            for h in rep.cause.hints:
                print(f"  hint  {h}")
    else:
        f = rep.first
        print("runs identical until")
        print()
        print(f"  step    {f.step}")
        print(f"  rank    {f.rank}")
        if f.module:
            print(f"  module  {f.module}")
        if f.phase:
            print(f"  phase   {f.phase}")
        print(f"  field   {f.kind}")
        print()
        print(f"  expected  {f.expected}")
        print(f"  observed  {f.observed}")
        print()
        print(f"cause  {rep.cause.name}")
        print(f"       {rep.cause.detail}")
        for h in rep.cause.hints:
            print(f"  hint  {h}")
        if rep.propagation:
            print()
            print("propagation  " + " -> ".join(rep.propagation))
        if rep.infected:
            print()
            for r, s in sorted(rep.infected.items()):
                print(f"rank {r} diverges from step {s}")

    for note in rep.notes:
        print(f"note  {note}")


def _show(run_dir: str, rank) -> None:
    run = Run.load(run_dir)
    h = run.header
    print(f"torch {h.get('torch')}  python {h.get('python')}")
    print(f"devices {', '.join(h.get('devices', []))}  world {h.get('world_size')}")
    det = h.get("determinism", {})
    on = [k for k, v in sorted(det.items()) if v]
    print(f"determinism  {', '.join(on) if on else 'none'}")
    print()
    ranks = [rank] if rank is not None else sorted(run.ranks)
    for r in ranks:
        recs = run.ranks.get(r, [])
        last = recs[-1] if recs else {}
        print(f"rank {r}  {len(recs)} steps  params {last.get('params', '-')}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="rewind",
        description="Find the first operation where two training runs diverge.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diff", help="compare two recorded runs")
    d.add_argument("run_a")
    d.add_argument("run_b")

    s = sub.add_parser("show", help="summarize a recorded run")
    s.add_argument("run")
    s.add_argument("--rank", type=int, default=None)

    args = p.parse_args(argv)
    if args.cmd == "diff":
        rep = diff_runs(args.run_a, args.run_b)
        _print_report(rep)
        return 0 if rep.identical else 1
    _show(args.run, args.rank)
    return 0


if __name__ == "__main__":
    sys.exit(main())
