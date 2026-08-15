import torch

from rewind import instability


def test_stable_function():
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    rep = instability(lambda: a @ b, runs=5)
    assert rep.stable
    assert rep.outputs == 1
    assert not rep.varying


def test_rng_consumption_is_not_instability():
    # dropout consumes RNG; with state restored around each run it is stable
    rep = instability(lambda: torch.dropout(torch.ones(64), 0.5, True), runs=5)
    assert rep.stable


def test_varying_function_detected():
    # without RNG restore, fresh draws differ between runs
    rep = instability(lambda: torch.rand(8), runs=5, restore_rng=False)
    assert not rep.stable
    assert rep.varying == [0]
    assert len(rep.digests[0]) > 1


def test_multiple_outputs():
    a = torch.randn(4, 4)
    rep = instability(lambda: (a.sum(), a * 2), runs=3)
    assert rep.stable
    assert rep.outputs == 2


def test_report_str():
    a = torch.ones(4)
    rep = instability(lambda: a, runs=2)
    assert "stable" in str(rep)
