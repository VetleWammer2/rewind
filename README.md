# rewind

Find the first operation where two supposedly identical training runs diverge.

```
$ rewind diff runs/a runs/b

runs identical until

  step    1842
  rank    6
  module  model.embed_tokens.weight
  phase   grad
  field   module

  expected  91fb73a0c44e21d7
  observed  ab810239de77f102

cause  fp-atomic
       Embedding uses nondeterministic atomics by default
  hint  torch.use_deterministic_algorithms(True) selects a deterministic implementation for this op

propagation  model.embed_tokens.weight -> params
```

Same seed does not mean same run. Floating-point atomics, autotuners, NCCL
scheduling and silent hardware faults all make runs diverge, and it usually
surfaces days later as a loss gap nobody can explain. rewind records cheap
bitwise fingerprints while you train, so the divergence can be located after
the fact instead of guessed at.

## Install

```
pip install git+https://github.com/VetleWammer2/rewind
```

## Use

```python
import rewind

rec = rewind.attach(model, optimizer, run_dir="runs/a")

for batch in loader:
    loss = model(batch).loss
    rec.mark("loss", loss)
    loss.backward()
    optimizer.step()        # closes the step
    optimizer.zero_grad()
rec.close()
```

Record both runs, then:

```
rewind diff runs/a runs/b
rewind show runs/a
```

To confirm a suspected nondeterministic op, re-run it on identical inputs:

```python
rewind.instability(lambda: module(x), runs=50)
# UNSTABLE over 50 runs: outputs [0] vary
```

## What it records

Per step and rank: module output and gradient fingerprints in execution
order, parameter and optimizer-state digests, RNG state, sample indices and
named probes. One JSONL ledger per rank plus a header with every
determinism-relevant flag, env var and version, so `diff` catches config
drift before touching numerics.

Fingerprints are order-sensitive 64-bit digests computed on the tensor's
device; only 8 bytes per tensor leave the GPU. v0.1 hashes synchronously at
hook time — fine for debugging runs, not yet tuned for always-on production
recording.

## Status

Works: recording, `diff` with root-cause classes (config-drift, fp-atomic,
rng-desync, dataloader, control-flow), instability check, CLI.

Roadmap: async side-stream hashing, NCCL collective schedule capture,
`rewind replay --step N --rank R` from checkpoints, op-level capture.

Design notes in [docs/design.md](docs/design.md).

## License

MIT
