import torch
import torch.nn as nn

import rewind
from rewind.cli import main as cli_main


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


_MAX_STEPS = 16


def _data(steps):
    # fixed dataset independent of run length, no RNG involved
    x = torch.linspace(-1, 1, _MAX_STEPS * 4 * 8).reshape(_MAX_STEPS, 4, 8)
    y = torch.linspace(1, -1, _MAX_STEPS * 4 * 4).reshape(_MAX_STEPS, 4, 4)
    return x[:steps], y[:steps]


def _train(run_dir, seed, steps=8, perturb_at=None, extra_rng_at=None):
    torch.manual_seed(seed)
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x, y = _data(steps)
    rec = rewind.attach(model, opt, str(run_dir))
    loss_fn = nn.MSELoss()
    for i in range(steps):
        if perturb_at == i:
            with torch.no_grad():
                model.fc1.weight.add_(1e-3)
        if extra_rng_at == i:
            torch.rand(1)
        out = model(x[i])
        loss = loss_fn(out, y[i])
        rec.set_batch(range(i * 4, i * 4 + 4))
        rec.mark("loss", loss)
        opt.zero_grad()
        loss.backward()
        opt.step()  # closes the step via the post hook
    rec.close()
    return run_dir


def test_identical_runs(tmp_path):
    _train(tmp_path / "a", seed=7)
    _train(tmp_path / "b", seed=7)
    rep = rewind.diff_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert rep.identical
    assert rep.first is None
    assert rep.cause.name == "none"


def test_different_seed_diverges_at_step_zero(tmp_path):
    _train(tmp_path / "a", seed=1)
    _train(tmp_path / "b", seed=2)
    rep = rewind.diff_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert not rep.identical
    assert rep.first.step == 0
    assert rep.first.kind == "module"
    assert rep.first.module == "fc1"
    assert rep.first.phase == "fwd"
    assert rep.first.expected != rep.first.observed


def test_perturbation_located(tmp_path):
    _train(tmp_path / "a", seed=7)
    _train(tmp_path / "b", seed=7, perturb_at=5)
    rep = rewind.diff_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert rep.first.step == 5
    assert rep.first.kind == "module"
    assert rep.first.module == "fc1"
    # divergence flows through the rest of the step
    assert "loss" in rep.propagation
    assert "params" in rep.propagation


def test_rng_desync_classified(tmp_path):
    _train(tmp_path / "a", seed=7)
    _train(tmp_path / "b", seed=7, extra_rng_at=3)
    rep = rewind.diff_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    # no tensor ever differs; only the RNG state does
    assert rep.first.step == 3
    assert rep.first.kind == "rng"
    assert rep.cause.name == "rng-desync"


def test_run_length_mismatch_noted(tmp_path):
    _train(tmp_path / "a", seed=7, steps=8)
    _train(tmp_path / "b", seed=7, steps=6)
    rep = rewind.diff_runs(str(tmp_path / "a"), str(tmp_path / "b"))
    assert rep.identical
    assert any("lengths differ" in n for n in rep.notes)


def test_cli_diff(tmp_path, capsys):
    _train(tmp_path / "a", seed=7)
    _train(tmp_path / "b", seed=7, perturb_at=2)
    code = cli_main(["diff", str(tmp_path / "a"), str(tmp_path / "b")])
    out = capsys.readouterr().out
    assert code == 1
    assert "runs identical until" in out
    assert "step    2" in out
    assert "fc1" in out


def test_cli_diff_identical(tmp_path, capsys):
    _train(tmp_path / "a", seed=7)
    _train(tmp_path / "b", seed=7)
    code = cli_main(["diff", str(tmp_path / "a"), str(tmp_path / "b")])
    assert code == 0
    assert "runs identical" in capsys.readouterr().out


def test_cli_show(tmp_path, capsys):
    _train(tmp_path / "a", seed=7)
    code = cli_main(["show", str(tmp_path / "a")])
    out = capsys.readouterr().out
    assert code == 0
    assert "rank 0" in out
    assert "8 steps" in out


def test_manual_step_without_optimizer(tmp_path):
    torch.manual_seed(3)
    model = Tiny()
    x, _ = _data(2)
    rec = rewind.attach(model, run_dir=str(tmp_path / "m"))
    for i in range(2):
        model(x[i])
        rec.step()
    rec.close()
    run = rewind.diff_runs(str(tmp_path / "m"), str(tmp_path / "m"))
    assert run.identical
