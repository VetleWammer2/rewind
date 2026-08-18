"""Clean-install smoke test for the README's CPU record/diff/show workflow."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch import nn

import rewind

_DIGEST = re.compile(r"[0-9a-f]{16}")


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def _record_run(run_dir: Path, order: tuple[int, ...]) -> None:
    torch.manual_seed(7)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    inputs = torch.tensor(
        [[0.25, -0.5], [0.75, 0.125], [-0.25, 1.0]], dtype=torch.float32
    )
    targets = torch.tensor([[0.5], [-0.25], [0.75]], dtype=torch.float32)
    loss_fn = nn.MSELoss()

    recorder = rewind.attach(model, optimizer, run_dir=str(run_dir))
    try:
        for sample_index in order:
            output = model(inputs[sample_index].unsqueeze(0))
            loss = loss_fn(output, targets[sample_index].unsqueeze(0))
            recorder.set_batch([sample_index])
            recorder.mark("loss", loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    finally:
        recorder.close()


def _require_digest(value: object, location: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AssertionError(f"{location} is not a 64-bit hexadecimal digest: {value!r}")


def _validate_run(
    run_dir: Path, expected_batches: tuple[int, ...]
) -> list[dict[str, object]]:
    files = {path.name for path in run_dir.iterdir()}
    if files != {"header.json", "rank0.jsonl"}:
        raise AssertionError(f"unexpected run files in {run_dir}: {sorted(files)}")

    with (run_dir / "header.json").open(encoding="utf-8") as stream:
        header = json.load(stream)
    required_header = {
        "rewind",
        "time",
        "torch",
        "python",
        "platform",
        "cuda",
        "cudnn",
        "devices",
        "world_size",
        "determinism",
        "env",
        "modules",
    }
    if not isinstance(header, dict) or not required_header <= header.keys():
        raise AssertionError(f"invalid header structure: {header!r}")
    if header["devices"] != ["cpu"] or header["world_size"] != 1:
        raise AssertionError(f"smoke test did not record a single CPU rank: {header!r}")
    if not isinstance(header["determinism"], dict):
        raise AssertionError("header determinism field is not an object")
    if not isinstance(header["modules"], dict) or "linear" not in header["modules"]:
        raise AssertionError("header does not describe the recorded model")

    with (run_dir / "rank0.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != len(expected_batches):
        raise AssertionError(f"expected {len(expected_batches)} steps, got {len(records)}")

    for step, (record, batch) in enumerate(zip(records, expected_batches)):
        if record.get("step") != step or record.get("batch") != [batch]:
            raise AssertionError(f"invalid logical step {step}: {record!r}")

        modules = record.get("modules")
        if not isinstance(modules, list) or not modules:
            raise AssertionError(f"step {step} has no module fingerprints")
        for position, entry in enumerate(modules):
            if (
                not isinstance(entry, list)
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or entry[1] not in {"fwd", "grad"}
            ):
                raise AssertionError(f"invalid module entry at step {step}: {entry!r}")
            _require_digest(entry[2], f"step {step} module {position}")

        marks = record.get("marks")
        if not isinstance(marks, dict) or set(marks) != {"loss"}:
            raise AssertionError(f"invalid marks at step {step}: {marks!r}")
        _require_digest(marks["loss"], f"step {step} loss")
        _require_digest(record.get("params"), f"step {step} params")
        _require_digest(record.get("optim"), f"step {step} optimizer")

        params_detail = record.get("params_detail")
        if not isinstance(params_detail, dict) or not params_detail:
            raise AssertionError(f"missing parameter details at step {step}")
        for name, digest in params_detail.items():
            _require_digest(digest, f"step {step} parameter {name}")

        rng = record.get("rng")
        if not isinstance(rng, dict) or set(rng) != {"cpu"}:
            raise AssertionError(f"invalid CPU RNG state at step {step}: {rng!r}")
        _require_digest(rng["cpu"], f"step {step} CPU RNG")

    return records


def _run_cli(
    executable: str, arguments: list[str], working_directory: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return subprocess.run(
        [executable, *arguments],
        cwd=working_directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_match(pattern: str, output: str, description: str) -> re.Match[str]:
    match = re.search(pattern, output, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing {description} in CLI output:\n{output}")
    return match


def _validate_cli(
    executable: str, working_directory: Path, final_params: str
) -> None:
    diff = _run_cli(executable, ["diff", "runs/a", "runs/b"], working_directory)
    if diff.returncode != 1:
        raise AssertionError(
            "divergent diff must return 1; "
            f"got {diff.returncode}\nstdout:\n{diff.stdout}\nstderr:\n{diff.stderr}"
        )
    _require_match(r"^  step\s+1$", diff.stdout, "first divergent step")
    _require_match(r"^  rank\s+0$", diff.stdout, "origin rank")
    _require_match(r"^  field\s+batch$", diff.stdout, "divergence field")
    _require_match(r"^  expected\s+\[1\]$", diff.stdout, "expected batch")
    _require_match(r"^  observed\s+\[2\]$", diff.stdout, "observed batch")
    _require_match(r"^cause\s+dataloader$", diff.stdout, "expected classification")

    show = _run_cli(executable, ["show", "runs/a"], working_directory)
    if show.returncode != 0:
        raise AssertionError(
            "show must return 0; "
            f"got {show.returncode}\nstdout:\n{show.stdout}\nstderr:\n{show.stderr}"
        )
    _require_match(
        r"^torch\s+\S+\s+python\s+\d+(?:\.\d+){2}$",
        show.stdout,
        "torch/python header",
    )
    _require_match(r"^devices\s+cpu\s+world\s+1$", show.stdout, "CPU world")
    summary = _require_match(
        r"^rank\s+(\d+)\s+(\d+)\s+steps\s+params\s+([0-9a-f]{16})$",
        show.stdout,
        "rank summary",
    )
    if summary.groups()[:2] != ("0", "3"):
        raise AssertionError(f"unexpected rank summary: {summary.group(0)!r}")
    if summary.group(3) != final_params:
        raise AssertionError("show did not report the final recorded parameter digest")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    package_file = Path(rewind.__file__).resolve()
    if package_file.parent == repository_root / "rewind":
        raise AssertionError(
            f"rewind imported from the checkout instead of an install: {package_file}"
        )
    if importlib.metadata.version("rewind") != rewind.__version__:
        raise AssertionError("installed package metadata and imported version disagree")

    executable = shutil.which("rewind")
    if executable is None:
        raise AssertionError("installed rewind console entry point is not on PATH")

    with tempfile.TemporaryDirectory(prefix="rewind-cpu-smoke-") as directory:
        working_directory = Path(directory)
        run_a = working_directory / "runs" / "a"
        run_b = working_directory / "runs" / "b"
        _record_run(run_a, (0, 1, 2))
        _record_run(run_b, (0, 2, 1))
        records_a = _validate_run(run_a, (0, 1, 2))
        records_b = _validate_run(run_b, (0, 2, 1))

        for field in ("modules", "marks", "params", "optim"):
            if records_a[1][field] == records_b[1][field]:
                raise AssertionError(f"step 1 {field} fingerprints did not diverge")

        report = rewind.diff_runs(str(run_a), str(run_b))
        first = report.first
        if (
            report.identical
            or report.config_diff
            or report.steps_compared != {0: 3}
            or first is None
            or (first.step, first.rank, first.kind) != (1, 0, "batch")
            or (first.expected, first.observed) != ("[1]", "[2]")
            or report.cause.name != "dataloader"
            or not {"loss", "params", "optimizer"} <= set(report.propagation)
        ):
            raise AssertionError(f"unexpected deterministic diff report: {report!r}")

        final_params = records_a[-1]["params"]
        if not isinstance(final_params, str):
            raise AssertionError("final parameter digest is not a string")
        _validate_cli(executable, working_directory, final_params)

    print("CPU README smoke test passed: record, diff, show, and ledger structure")


if __name__ == "__main__":
    main()
