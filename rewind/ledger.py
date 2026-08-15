"""Run ledger: header.json plus one append-only JSONL file per rank."""

import glob
import json
import os
import re


class LedgerWriter:
    def __init__(self, run_dir: str, rank: int):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.rank = rank
        self._f = open(os.path.join(run_dir, f"rank{rank}.jsonl"), "w", encoding="utf-8")

    def write_header(self, header: dict) -> None:
        path = os.path.join(self.run_dir, "header.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(header, f, indent=1, sort_keys=True)

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


class Run:
    """A recorded run loaded from disk."""

    def __init__(self, header: dict, ranks: dict):
        self.header = header
        self.ranks = ranks  # rank -> list of step records

    @classmethod
    def load(cls, run_dir: str) -> "Run":
        header_path = os.path.join(run_dir, "header.json")
        if not os.path.isfile(header_path):
            raise FileNotFoundError(f"not a rewind run (no header.json): {run_dir}")
        with open(header_path, encoding="utf-8") as f:
            header = json.load(f)
        ranks = {}
        for path in sorted(glob.glob(os.path.join(run_dir, "rank*.jsonl"))):
            m = re.search(r"rank(\d+)\.jsonl$", path)
            if not m:
                continue
            records = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            ranks[int(m.group(1))] = records
        if not ranks:
            raise FileNotFoundError(f"no rank ledgers in {run_dir}")
        return cls(header, ranks)
