# Design notes

## Problem

Two runs with identical code, data and seeds still diverge. Sources, roughly
in order of how often they bite:

1. **config-drift** — env/flag/version differences between the runs.
2. **fp-atomic** — CUDA kernels using floating-point atomics; addition is not
   associative, thread order varies. Two families: ops with a deterministic
   implementation behind `torch.use_deterministic_algorithms(True)`
   (scatter_add, index_add, Embedding backward, conv, ...) and ops with none
   at all (CTCLoss backward, cumsum on float, EmbeddingBag max backward,
   interpolate backward, ...). The second family cannot be flagged away.
3. **algo-autotune** — `cudnn.benchmark` and inductor max-autotune pick
   kernels by on-device timing; noise picks different kernels per run.
4. **nccl-reduction-order** — NCCL selects algorithm/protocol from detected
   topology and message size. Selection shifts with version, env, NIC state
   or rack layout, and shifts reduction order with it. Env vars are not
   ground truth: NCCL falls back silently.
5. **rng-desync** — one run consumes more or fewer random numbers. CUDA RNG
   is Philox (seed, offset); an offset delta pinpoints the extra call.
6. **dataloader** — worker scheduling, seeding, sampler state.
7. **sdc** — silent data corruption from defective hardware. Documented at
   fleet scale by Google and Meta; loss curves can look healthy while
   corruption steers optimization.

Standard practice when a run diverges is to rewind to a checkpoint, skip
batches and hope. The goal here is to name the first divergent operation
instead.

## Approach

Record cheap fingerprints during training; do all analysis offline.

**Fingerprints.** Order-sensitive 64-bit digests of a tensor's bit pattern,
computed on the tensor's device. Each 64-bit word is salted with its position
before folding — plain XOR is permutation-invariant and cancels paired bit
flips, which is exactly the failure mode a divergence tool cannot afford.
Dtype and shape are salted in. Only the 8-byte digest crosses to the host,
batched once per step.

**Capture tiers.** Cost dictates granularity:

- step level: params/optimizer roots, RNG, dataloader position — always fine
- module level: forward output + gradient digests via hooks — default
- op level: dispatch interception — replay/triage only, never always-on

**Ledger.** One append-only JSONL file per rank plus `header.json` with
every determinism-relevant flag, prefixed env var, version and the module
name→class map. Durable and step-indexed: a divergence at step 1842 must
still be there when you look for it.

**Diff.** Phases, cheapest first:

1. header diff → config-drift, reported before any tensor work
2. per-rank scan for the first step whose record differs
3. within that step, walk the recorded execution order (batch → module
   fwd/grad stream → marks → params → optimizer → RNG) to the first
   divergent entry. A mismatch in the stream itself (different module
   sequence) is control-flow divergence, reported as such rather than
   crashing the aligner.
4. classify against the taxonomy above; report the propagation chain (which
   later modules, loss, params also diverge) and which ranks diverge at
   later steps (infected via collectives).

Alignment is on logical time — step count, execution index — never wall
clock.

**Instability check.** The confirmation step. Restore RNG state, run the
suspect computation N times on identical inputs, compare digests. Varying
digests on identical state is nondeterminism on this hardware, full stop.
RNG state is restored around every run so ordinary RNG consumption does not
read as instability.

## Replay (roadmap)

`rewind replay --step N --rank R` is checkpoint-and-reexecute: restore the
nearest checkpoint plus a sidecar (full Philox state blobs — not seeds —
GradScaler after update(), dataloader state), fast-forward to N, re-execute
the step with op-level capture.

Two honest problems shape the design:

- Exact replay presupposes the determinism you are debugging the absence of.
  So fast-forward is verified, not assumed: every step's fingerprints are
  compared against the ledger, and a mismatch during fast-forward *is* the
  finding, not a failure.
- Multi-rank determinism is much harder than single-rank. Stubbing
  collectives with recorded incoming payloads reduces the requirement to
  single-rank compute determinism. That is the plan; nobody ships it today.

Process-snapshot replay (rr-style) is off the table for GPUs: proprietary
ioctls, DMA past the syscall boundary, hardware nondeterminism.

## What this is not

Flight Recorder debugs hangs from collective metadata; it records no tensor
values and compares ranks within one run, not runs. Binary instrumentation
tools (NVBit-class) trace instructions inside kernels at 1.5–100x cost —
the layer below this one, useful after rewind names the kernel. Invariant
checkers catch within-run bugs that diverge from *correctness*; rewind
catches runs that diverge from *each other* — a bug both runs share
produces zero diff. Experiment trackers compare scalar metrics, which is
how you notice divergence, not how you locate it.

## Overhead

v0.1 hashes synchronously in hooks. Hashing is memory-bandwidth-bound, so
module-boundary digests cost roughly one extra read of the activations —
acceptable for debugging runs on small and mid models, not yet for always-on
recording of large ones. The async path (side-stream hashing, digest drain
once per step, every-N-step weight roots) is the roadmap item that makes
always-on realistic; the ledger format already assumes it.
