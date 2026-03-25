use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_dft::Radix2DFTSmallBatch;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::DenseMatrix;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::time::Duration;
use whir_p3::whir::bench_support::{BaseDftBenchmarkBackend, backend_name, run_padded_base_dft};
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
use whir_p3::whir::bench_support::{
    metal_available, prepare_metal_kernel_only_base_dft, run_metal_kernel_only_base_dft,
};

type F = KoalaBear;

#[derive(Debug, Clone, Copy)]
struct DftCase {
    name: &'static str,
    width: usize,
    fft_size: usize,
}

impl DftCase {
    #[must_use]
    const fn element_count(self) -> usize {
        self.width * self.fft_size
    }
}

fn generate_padded_matrix(case: DftCase, seed: u64) -> DenseMatrix<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let values = (0..case.element_count()).map(|_| rng.random()).collect();
    DenseMatrix::new(values, case.width)
}

fn benchmark_padded_base_dft(c: &mut Criterion) {
    let mut group = c.benchmark_group("dft_padded_base");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    let cases = [
        DftCase {
            name: "tiny_1x256",
            width: 1,
            fft_size: 1 << 8,
        },
        DftCase {
            name: "small_4x4096",
            width: 4,
            fft_size: 1 << 12,
        },
        DftCase {
            name: "whir_round3_16x32768",
            width: 16,
            fft_size: 1 << 15,
        },
        DftCase {
            name: "whir_round2_16x65536",
            width: 16,
            fft_size: 1 << 16,
        },
        DftCase {
            name: "whir_round1_16x131072",
            width: 16,
            fft_size: 1 << 17,
        },
        DftCase {
            name: "whir_round0_16x262144",
            width: 16,
            fft_size: 1 << 18,
        },
    ];

    #[cfg(not(all(feature = "gpu-metal", target_os = "macos")))]
    let backends = vec![BaseDftBenchmarkBackend::Cpu];

    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    let backends = {
        let mut backends = vec![BaseDftBenchmarkBackend::Cpu];
        if metal_available() {
            backends.push(BaseDftBenchmarkBackend::Metal);
        } else {
            std::eprintln!(
                "gpu-metal enabled, but Metal is unavailable to this process; skipping metal DFT benchmarks"
            );
        }
        backends
    };

    for (idx, case) in cases.into_iter().enumerate() {
        let padded = generate_padded_matrix(case, 0xD3F7_0000_u64.wrapping_add(idx as u64));
        let dft = Radix2DFTSmallBatch::<F>::new(case.fft_size);
        group.throughput(Throughput::Elements(case.element_count() as u64));

        for backend in backends.iter().copied() {
            let bench_name = format!("{}/{}", backend_name(backend), case.name);
            group.bench_with_input(
                BenchmarkId::from_parameter(bench_name),
                &case,
                |b, &_case| {
                    b.iter_batched(
                        || padded.clone(),
                        |padded| {
                            let _ = run_padded_base_dft(backend, &dft, padded);
                        },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    group.finish();
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
fn benchmark_metal_kernel_only_base_dft(c: &mut Criterion) {
    if !metal_available() {
        std::eprintln!(
            "gpu-metal enabled, but Metal is unavailable to this process; skipping dispatch-only metal DFT benchmarks"
        );
        return;
    }

    let mut group = c.benchmark_group("dft_padded_base_metal_kernel_only");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    let cases = [
        DftCase {
            name: "tiny_1x256",
            width: 1,
            fft_size: 1 << 8,
        },
        DftCase {
            name: "small_4x4096",
            width: 4,
            fft_size: 1 << 12,
        },
        DftCase {
            name: "whir_round3_16x32768",
            width: 16,
            fft_size: 1 << 15,
        },
        DftCase {
            name: "whir_round2_16x65536",
            width: 16,
            fft_size: 1 << 16,
        },
        DftCase {
            name: "whir_round1_16x131072",
            width: 16,
            fft_size: 1 << 17,
        },
        DftCase {
            name: "whir_round0_16x262144",
            width: 16,
            fft_size: 1 << 18,
        },
    ];

    for (idx, case) in cases.into_iter().enumerate() {
        let padded = generate_padded_matrix(case, 0xD3F7_1000_u64.wrapping_add(idx as u64));
        let Some(prepared) = prepare_metal_kernel_only_base_dft(&padded) else {
            std::eprintln!(
                "failed to prepare dispatch-only Metal benchmark for {}",
                case.name
            );
            continue;
        };
        group.throughput(Throughput::Elements(case.element_count() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(case.name),
            &case,
            |b, &_case| {
                b.iter(|| {
                    let ran = run_metal_kernel_only_base_dft(&prepared);
                    assert!(
                        ran,
                        "dispatch-only Metal benchmark failed for {}",
                        case.name
                    );
                });
            },
        );
    }

    group.finish();
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
criterion_group!(
    benches,
    benchmark_padded_base_dft,
    benchmark_metal_kernel_only_base_dft
);
#[cfg(not(all(feature = "gpu-metal", target_os = "macos")))]
criterion_group!(benches, benchmark_padded_base_dft);
criterion_main!(benches);
