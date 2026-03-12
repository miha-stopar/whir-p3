# GPU Optimization Plan (Baseline + Roadmap)

## Branch

- `gpu-optimization-plan`

## Baseline Benchmark Runs

All runs were executed with:

- `cargo +1.93 bench -- --sample-size 10`
- Rust toolchain `1.93.1` (installed because this repo requires `rustc >= 1.93`)

### 1) `whir`

Command:

```bash
cargo +1.93 bench --bench whir -- --sample-size 10
```

Key results:

- `commit`: `363.51 ms .. 429.49 ms`
- `prove`: `880.32 ms .. 1.2277 s`

### 2) `sumcheck` (Classic only)

The unfiltered benchmark panics in SVO mode in this branch, so Classic was run explicitly.

Command:

```bash
cargo +1.93 bench --bench sumcheck -- --sample-size 10 Classic
```

Key results:

- `SumcheckProver/Classic/16`: `1.2997 ms .. 4.4827 ms`
- `SumcheckProver/Classic/18`: `3.1118 ms .. 3.2188 ms`
- `SumcheckProver/Classic/20`: `8.8973 ms .. 13.110 ms`
- `SumcheckProver/Classic/22`: `43.689 ms .. 49.865 ms`
- `SumcheckProver/Classic/24`: `124.48 ms .. 137.23 ms`

Observed issue during unfiltered run:

- `cargo +1.93 bench --bench sumcheck -- --sample-size 10`
- Panic at `src/sumcheck/sumcheck_prover.rs:258` (`assertion left == right`, left `2`, right `8`)

### 3) `evaluate`

Command:

```bash
cargo +1.93 bench --bench evaluate -- --sample-size 10
```

Key results:

- `evaluate/16`: `127.29 us .. 133.31 us`
- `evaluate/20`: `473.92 us .. 498.03 us`
- `evaluate/25`: `7.8090 ms .. 7.9168 ms`

### 4) `eval_multilinear`

Command:

```bash
cargo +1.93 bench --bench eval_multilinear -- --sample-size 10
```

Representative results:

- `eval_multilinear_base/packed-split/22`: `6.5744 ms .. 10.640 ms`
- `eval_multilinear_ext/packed-split/22`: `14.860 ms .. 15.075 ms`

### 5) `stir_queries`

Command:

```bash
cargo +1.93 bench --bench stir_queries -- --sample-size 10
```

Representative results:

- `benchmark main round 1`: `20.594 us .. 21.972 us`
- `very_large_256_queries_1m_domain`: `36.751 us .. 41.091 us`

## GPU Roadmap

## Phase 1 (highest ROI): GPU DFT path

Targets:

- `src/whir/committer/writer.rs` (`dft_batch` in commitment phase)
- `src/whir/prover/mod.rs` (`dft_algebra_batch` in each proving round)

Why first:

- These kernels are large, regular, and dominate commit/round work.
- Lowest protocol risk: they are cleanly behind DFT interfaces.

Plan:

1. Implement a GPU-backed DFT adapter (Metal first on Apple Silicon).
2. Keep CPU fallback and runtime feature flag.
3. Add parity tests CPU vs GPU outputs.

Success criteria:

- >= 1.5x speedup on `bench whir` `commit`.
- No proof/verifier divergence.

### Phase 1 Context: What the DFT is doing in WHIR

At commitment time and at every prover round, WHIR must re-encode the current folded polynomial
over a larger Reed-Solomon domain before building a Merkle commitment. In this codebase, that
re-encoding step is the batched DFT.

- Commitment path: `src/whir/committer/writer.rs` (`dft_batch`)
- Prover-round path: `src/whir/prover/mod.rs` (`dft_algebra_batch`)

The inputs are currently 1D evaluation vectors (`2^n` values). The code reshapes them into a
matrix, transposes it, pads rows (to match the RS blowup / inverse rate), and then runs a DFT per
column.

### Data layout visualization

Given current variable count `n_r` and next folding factor `k_next`:

```text
1D vector length = 2^n_r
      |
reshape with width = 2^(n_r - k_next)
      |
pre-transpose matrix:  rows = 2^k_next,      cols = 2^(n_r-k_next)
      |
transpose
      v
post-transpose matrix: rows = 2^(n_r-k_next), cols = 2^k_next
      |
pad rows to inv_rate(r) * 2^(n_r-k_next)
      v
run batched DFT over columns
```

Interpretation of dimensions after transpose:

- `width = 2^k_next` is the number of independent FFT streams (batch count).
- `height` (after padding) is the FFT size for each stream.

So the batched DFT workload is:

```text
batch_count FFTs, each of length fft_size
where:
  batch_count = width = 2^k_next
  fft_size    = padded_height
```

### Concrete numbers (from `benches/whir.rs` defaults)

Benchmark defaults:

- `num_variables = 24`
- `folding_factor = Constant(4)` (so `k_next = 4` in all rounds)
- `starting_log_inv_rate = 1`
- `rs_domain_initial_reduction_factor = 3`

#### Commitment phase

- `n = 24`, `k0 = 4`
- post-transpose width = `2^4 = 16`
- post-transpose base height = `2^(24-4) = 2^20 = 1,048,576`
- padded height = `2^(24 + 1 - 4) = 2^21 = 2,097,152`

Workload: `16` FFTs of size `2,097,152`.

#### Prover rounds

| Round | `n_r` | Width (`2^k`) | Base Height (`2^(n_r-k)`) | `inv_rate(r)` | Padded Height | DFT Workload |
|---|---:|---:|---:|---:|---:|---|
| 0 | 20 | 16 | 65,536 | 4 | 262,144 | 16 x 262,144 |
| 1 | 16 | 16 | 4,096 | 32 | 131,072 | 16 x 131,072 |
| 2 | 12 | 16 | 256 | 256 | 65,536 | 16 x 65,536 |
| 3 | 8 | 16 | 16 | 2,048 | 32,768 | 16 x 32,768 |
| 4 | 4 | 16 | 1 | 16,384 | 16,384 | 16 x 16,384 |

This is why Phase 1 starts with DFT offload: these are regular, repeated, high-throughput kernels
with stable shapes and low protocol-risk integration points.

## Phase 2: GPU multilinear kernels used by sumcheck

Targets:

- `src/poly/evals.rs`
- `evaluate_hypercube_base` / `evaluate_hypercube_ext`
- `sumcheck_coefficients`
- `compress` / `compress_into_packed`

Why second:

- Sumcheck cost grows quickly with variable count.
- Kernels are data-parallel but have more reduction/detail work than DFT.

Plan:

1. Offload vectorized fold/reduction kernels.
2. Minimize host-device transfers by batching round work.
3. Use deterministic reduction strategy to avoid transcript mismatches.

Success criteria:

- >= 1.5x speedup on `sumcheck` Classic (20-24 vars).

## Phase 3 (optional): GPU Merkle hashing pipeline

Targets:

- Merkle commitment/opening path around `commit_matrix` and `open_batch`.

Why optional:

- Potentially large win, but higher complexity and crypto-kernel engineering risk.

Plan:

1. Prototype leaf hashing on GPU.
2. Keep compression/hash compatibility exact with CPU implementation.
3. Validate roots and opening proofs bit-for-bit.

Success criteria:

- Additional speedup on `whir` `commit` and `prove` without changing proof bytes.

## Guardrails

- Keep `CPU` as default backend until GPU passes correctness + benchmark gates.
- Require deterministic behavior and stable transcript compatibility.
- Add CI job running CPU and GPU parity tests on small/medium instances.

## Current Implementation Status

- DFT shape math is centralized in `src/whir/dft_layout.rs`.
- Both DFT callsites (`commit` and prover rounds) route through backend hooks in
  `src/whir/dft_backend.rs`.
- Backend paths are split:
  - `gpu-metal` feature for Metal path
  - `gpu-vulkan` feature for Vulkan path
- Current GPU paths are intentionally CPU fallbacks until kernels are implemented.

Build matrix currently validated:

- `cargo check`
- `cargo check --features gpu-metal`
- `cargo check --features gpu-vulkan`
- `cargo check --features "gpu-metal,gpu-vulkan"`
