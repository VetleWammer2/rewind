"""Root-cause classification of a located divergence.

Class names follow docs/design.md. Module-level matching is a heuristic:
naming the exact aten op needs op-level capture (roadmap).
"""

from dataclasses import dataclass, field

# CUDA modules whose backward uses atomics but has a deterministic
# implementation behind torch.use_deterministic_algorithms(True).
# https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
SWITCHABLE = {
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "Embedding",
    "MaxPool3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
}

# CUDA modules with no deterministic implementation at all: the flag makes
# them raise instead. These cannot be flagged away.
NO_DETERMINISTIC_IMPL = {
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveMaxPool2d",
    "AvgPool3d",
    "CTCLoss",
    "EmbeddingBag",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "NLLLoss",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "Upsample",
    "UpsamplingBilinear2d",
}


@dataclass
class Cause:
    name: str
    detail: str
    hints: list = field(default_factory=list)


def classify(divergence, config_diff: dict, header_a: dict, header_b: dict) -> Cause:
    hints = []
    det = (header_a or {}).get("determinism", {})
    on_gpu = (header_a or {}).get("devices", ["cpu"]) != ["cpu"]

    if config_diff:
        keys = ", ".join(sorted(config_diff))
        return Cause(
            "config-drift",
            f"runs were configured differently: {keys}",
            ["fix the config diff before chasing numerics"],
        )

    if divergence is None:
        return Cause("none", "runs are identical over the compared steps")

    if divergence.kind == "batch":
        return Cause(
            "dataloader",
            "the runs consumed different samples at this step",
            ["check dataloader seeding, worker count and sampler state"],
        )

    if divergence.kind == "rng":
        return Cause(
            "rng-desync",
            "RNG state diverged without any tensor divergence: one run consumed "
            "more or fewer random numbers than the other",
            ["look for an extra/missing dropout, init or torch.rand call"],
        )

    if divergence.kind == "stream":
        return Cause(
            "control-flow",
            "the runs executed different module sequences at this step",
            ["check data-dependent branches, early exits and recompilation"],
        )

    if divergence.kind == "module":
        # grad-phase entries carry parameter names ("embed.weight");
        # resolve to the owning module before the class lookup
        mods = (header_a or {}).get("modules", {})
        name = divergence.module
        cls = mods.get(name, "")
        while not cls and "." in name:
            name = name.rsplit(".", 1)[0]
            cls = mods.get(name, "")
        if cls in NO_DETERMINISTIC_IMPL:
            return Cause(
                "fp-atomic",
                f"{cls} has no deterministic CUDA implementation; its "
                "atomics-based kernels are expected to diverge run-to-run",
                ["confirm with rewind.instability()", "no flag fixes this op"],
            )
        if cls in SWITCHABLE:
            return Cause(
                "fp-atomic",
                f"{cls} uses nondeterministic atomics by default",
                [
                    "torch.use_deterministic_algorithms(True) selects a "
                    "deterministic implementation for this op",
                    "confirm with rewind.instability()",
                ],
            )
        if on_gpu and det.get("cudnn_benchmark"):
            hints.append(
                "cudnn.benchmark is on: autotuning can pick different "
                "algorithms per run"
            )
        if on_gpu and not (header_a or {}).get("env", {}).get(
            "CUBLAS_WORKSPACE_CONFIG"
        ):
            hints.append(
                "CUBLAS_WORKSPACE_CONFIG is unset: multi-stream cuBLAS can "
                "be nondeterministic"
            )
        hints.append(
            "if rewind.instability() is stable on this step, suspect data "
            "or hardware (SDC) instead of kernel nondeterminism"
        )
        return Cause(
            "unknown",
            f"first divergence in {divergence.module} ({cls or 'unknown type'}), "
            f"phase {divergence.phase}",
            hints,
        )

    return Cause("unknown", f"first divergence in field '{divergence.kind}'", hints)
