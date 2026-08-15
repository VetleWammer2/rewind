from rewind.classify import classify
from rewind.diff import Divergence

HEADER = {
    "modules": {"embed": "Embedding", "fc": "Linear", "ctc": "CTCLoss"},
    "devices": ["NVIDIA H100"],
    "determinism": {"cudnn_benchmark": False},
    "env": {},
}


def test_grad_phase_param_resolves_to_module():
    d = Divergence(5, 0, "module", module="embed.weight", phase="grad")
    c = classify(d, {}, HEADER, HEADER)
    assert c.name == "fp-atomic"
    assert "Embedding" in c.detail


def test_no_deterministic_impl():
    d = Divergence(2, 0, "module", module="ctc", phase="fwd")
    c = classify(d, {}, HEADER, HEADER)
    assert c.name == "fp-atomic"
    assert "no deterministic" in c.detail


def test_config_drift_wins():
    d = Divergence(0, 0, "module", module="fc", phase="fwd")
    c = classify(d, {"env.NCCL_ALGO": ("Ring", None)}, HEADER, HEADER)
    assert c.name == "config-drift"


def test_unknown_module_gets_hints():
    d = Divergence(3, 0, "module", module="fc", phase="fwd")
    c = classify(d, {}, HEADER, HEADER)
    assert c.name == "unknown"
    assert c.hints


def test_rng_and_batch_kinds():
    assert classify(Divergence(1, 0, "rng"), {}, HEADER, HEADER).name == "rng-desync"
    assert classify(Divergence(1, 0, "batch"), {}, HEADER, HEADER).name == "dataloader"
    assert (
        classify(Divergence(1, 0, "stream", module="fc"), {}, HEADER, HEADER).name
        == "control-flow"
    )
