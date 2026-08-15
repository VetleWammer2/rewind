"""Recorder: hooks into a model/optimizer and writes a fingerprint ledger."""

import os
import platform
import time

import torch

from . import hashing
from .ledger import LedgerWriter

_ENV_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "CUDNN_",
    "NCCL_",
    "NVTE_",
    "PYTORCH_",
    "TORCH",
)


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_world() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from _iter_tensors(obj[k])


class Recorder:
    """Records per-step fingerprints of module outputs, gradients, parameters,
    optimizer state and RNG state to a per-rank ledger.

    A step is closed on every optimizer.step() (via a post hook), or manually
    with step() when no optimizer is given.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        run_dir: str = "runs/run",
        *,
        rank: int | None = None,
        param_detail_every: int = 1,
        grad_hashes: bool = True,
        module_filter=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.rank = _dist_rank() if rank is None else rank
        self.param_detail_every = param_detail_every
        self.step_idx = 0
        self._buf = []  # (name, phase, 0-dim digest tensor) in execution order
        self._marks = []  # (name, digest tensor)
        self._batch = None
        self._hooks = []
        self._closed = False

        self.writer = LedgerWriter(run_dir, self.rank)
        if self.rank == 0:
            self.writer.write_header(self._header())

        for fqn, mod in model.named_modules():
            if module_filter is not None and not module_filter(fqn, mod):
                continue
            name = fqn or type(model).__name__
            self._hooks.append(mod.register_forward_hook(self._make_fwd_hook(name)))
        if grad_hashes:
            for pname, p in model.named_parameters():
                if p.requires_grad:
                    self._hooks.append(
                        p.register_post_accumulate_grad_hook(
                            self._make_grad_hook(pname)
                        )
                    )
        if optimizer is not None:
            self._hooks.append(
                optimizer.register_step_post_hook(self._on_optimizer_step)
            )

    # -- capture ----------------------------------------------------------

    def _make_fwd_hook(self, name):
        def hook(mod, inputs, output):
            digests = [hashing.fingerprint(t) for t in _iter_tensors(output)]
            if digests:
                self._buf.append((name, "fwd", hashing.combine(digests)))

        return hook

    def _make_grad_hook(self, pname):
        def hook(param):
            if param.grad is not None:
                self._buf.append((pname, "grad", hashing.fingerprint(param.grad)))

        return hook

    def _on_optimizer_step(self, optimizer, args, kwargs):
        self.step()

    def mark(self, name: str, value) -> None:
        """Record a named probe (loss, custom tensors) for the current step."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self._marks.append((name, hashing.fingerprint(value)))

    def set_batch(self, indices) -> None:
        """Record which samples the current step consumed."""
        self._batch = [int(i) for i in indices]

    # -- step boundary ----------------------------------------------------

    def step(self) -> None:
        """Close the current step and write its ledger record."""
        rec = {"step": self.step_idx}
        if self._batch is not None:
            rec["batch"] = self._batch

        pending = [d for _, _, d in self._buf]
        pending += [d for _, d in self._marks]

        param_names = []
        for pname, p in self.model.named_parameters():
            pending.append(hashing.fingerprint(p))
            param_names.append(pname)
        n_optim = 0
        if self.optimizer is not None:
            for st in self.optimizer.state.values():
                for k in sorted(st, key=str):
                    if isinstance(st[k], torch.Tensor):
                        pending.append(hashing.fingerprint(st[k]))
                        n_optim += 1

        values = self._drain(pending)
        i = 0
        mod_vals = values[i : i + len(self._buf)]
        i += len(self._buf)
        mark_vals = values[i : i + len(self._marks)]
        i += len(self._marks)
        pvals = values[i : i + len(param_names)]
        i += len(param_names)
        ovals = values[i : i + n_optim]

        rec["modules"] = [
            [n, ph, hashing.hex_digest(v)]
            for (n, ph, _), v in zip(self._buf, mod_vals)
        ]
        if self._marks:
            rec["marks"] = {
                n: hashing.hex_digest(v) for (n, _), v in zip(self._marks, mark_vals)
            }

        rec["params"] = hashing.hex_digest(hashing.combine_ints(pvals))
        if self.param_detail_every and self.step_idx % self.param_detail_every == 0:
            rec["params_detail"] = {
                n: hashing.hex_digest(v) for n, v in zip(param_names, pvals)
            }
        if self.optimizer is not None:
            rec["optim"] = hashing.hex_digest(hashing.combine_ints(ovals))

        rec["rng"] = self._rng_state()

        self.writer.write(rec)
        self._buf = []
        self._marks = []
        self._batch = None
        self.step_idx += 1

    def _drain(self, digests: list) -> list:
        """Convert 0-dim digest tensors to ints with one transfer per device."""
        if not digests:
            return []
        by_dev = {}
        for i, d in enumerate(digests):
            by_dev.setdefault(d.device, []).append(i)
        out = [0] * len(digests)
        for dev, idxs in by_dev.items():
            vals = torch.stack([digests[i] for i in idxs]).cpu().tolist()
            for i, v in zip(idxs, vals):
                out[i] = v
        return out

    def _rng_state(self) -> dict:
        rng = {"cpu": hashing.hex_digest(hashing.fingerprint_int(torch.get_rng_state()))}
        if torch.cuda.is_available():
            rng["cuda"] = [
                hashing.hex_digest(hashing.fingerprint_int(s))
                for s in torch.cuda.get_rng_state_all()
            ]
        return rng

    # -- header -----------------------------------------------------------

    def _header(self) -> dict:
        try:
            cudnn_version = torch.backends.cudnn.version()
        except Exception:
            cudnn_version = None
        return {
            "rewind": _version(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "cudnn": cudnn_version,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else ["cpu"],
            "world_size": _dist_world(),
            "determinism": {
                "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            },
            "env": {
                k: v
                for k, v in sorted(os.environ.items())
                if k.startswith(_ENV_PREFIXES)
            },
            "modules": {
                fqn or type(self.model).__name__: type(m).__name__
                for fqn, m in self.model.named_modules()
            },
        }

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        for h in self._hooks:
            h.remove()
        self.writer.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def attach(model, optimizer=None, run_dir: str = "runs/run", **kwargs) -> Recorder:
    """Attach a Recorder to a model. Ledger goes to run_dir/rank<r>.jsonl."""
    return Recorder(model, optimizer, run_dir, **kwargs)


def _version() -> str:
    from . import __version__

    return __version__
