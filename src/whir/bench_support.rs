//! Hidden benchmark support helpers.
//!
//! These helpers expose explicit DFT backend selection to Criterion benches
//! without widening the normal WHIR API surface.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::dense::DenseMatrix;

use crate::whir::dft_backend::{
    DftBackend, run_padded_base_dft_explicit_cpu, with_explicit_backend_override,
};
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
use crate::whir::dft_backend::{
    prepare_padded_base_dft_dispatch_only_metal, run_padded_base_dft_dispatch_only_metal,
};
#[cfg(feature = "gpu-wgsl")]
use crate::whir::dft_backend::{run_padded_base_dft_explicit_wgsl, wgsl_is_available_for_bench};

/// Explicit backend choice for padded base-field DFT microbenchmarks.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseDftBenchmarkBackend {
    Cpu,
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    Metal,
    #[cfg(feature = "gpu-wgsl")]
    Wgsl,
}

/// Prepared dispatch-only Metal benchmark state.
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[doc(hidden)]
#[derive(Debug)]
pub struct MetalKernelOnlyBenchmark {
    inner: crate::whir::dft_backend::MetalDispatchOnlyBenchmark,
}

/// Return whether a usable Metal runtime is visible to the current process.
#[doc(hidden)]
#[must_use]
pub fn metal_available() -> bool {
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    {
        return crate::whir::dft_backend::metal_is_available_for_bench();
    }

    #[cfg(not(all(feature = "gpu-metal", target_os = "macos")))]
    {
        false
    }
}

/// Return whether a usable WGSL/WGPU runtime is visible to the current process.
#[cfg(feature = "gpu-wgsl")]
#[doc(hidden)]
#[must_use]
pub fn wgsl_available() -> bool {
    wgsl_is_available_for_bench()
}

/// Stable backend label for benchmark names.
#[doc(hidden)]
#[must_use]
pub const fn backend_name(backend: BaseDftBenchmarkBackend) -> &'static str {
    match backend {
        BaseDftBenchmarkBackend::Cpu => "cpu",
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        BaseDftBenchmarkBackend::Metal => "metal",
        #[cfg(feature = "gpu-wgsl")]
        BaseDftBenchmarkBackend::Wgsl => "wgsl",
    }
}

/// Return whether the selected benchmark backend is usable in this process.
#[doc(hidden)]
#[must_use]
pub fn backend_available(backend: BaseDftBenchmarkBackend) -> bool {
    match backend {
        BaseDftBenchmarkBackend::Cpu => true,
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        BaseDftBenchmarkBackend::Metal => metal_available(),
        #[cfg(feature = "gpu-wgsl")]
        BaseDftBenchmarkBackend::Wgsl => wgsl_available(),
    }
}

/// Execute a padded base-field DFT against an explicit backend.
#[doc(hidden)]
pub fn run_padded_base_dft<F, Dft>(
    backend: BaseDftBenchmarkBackend,
    dft: &Dft,
    padded: DenseMatrix<F>,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    match backend {
        BaseDftBenchmarkBackend::Cpu => run_padded_base_dft_explicit_cpu(dft, padded),
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        BaseDftBenchmarkBackend::Metal => {
            crate::whir::dft_backend::run_padded_base_dft_explicit_metal(dft, padded)
        }
        #[cfg(feature = "gpu-wgsl")]
        BaseDftBenchmarkBackend::Wgsl => run_padded_base_dft_explicit_wgsl(dft, padded),
    }
}

/// Execute a closure with an explicit backend override for full WHIR benches.
#[doc(hidden)]
pub fn with_benchmark_backend<R>(backend: BaseDftBenchmarkBackend, f: impl FnOnce() -> R) -> R {
    let backend = match backend {
        BaseDftBenchmarkBackend::Cpu => DftBackend::Cpu,
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        BaseDftBenchmarkBackend::Metal => DftBackend::Metal,
        #[cfg(feature = "gpu-wgsl")]
        BaseDftBenchmarkBackend::Wgsl => DftBackend::Wgsl,
    };
    with_explicit_backend_override(backend, f)
}

/// Prepare a dispatch-only Metal benchmark state with input already uploaded.
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[doc(hidden)]
pub fn prepare_metal_kernel_only_base_dft<F>(
    padded: &DenseMatrix<F>,
) -> Option<MetalKernelOnlyBenchmark>
where
    F: TwoAdicField,
{
    prepare_padded_base_dft_dispatch_only_metal(padded)
        .map(|inner| MetalKernelOnlyBenchmark { inner })
}

/// Run the already-uploaded Metal DFT benchmark state without host marshaling.
#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[doc(hidden)]
pub fn run_metal_kernel_only_base_dft(benchmark: &MetalKernelOnlyBenchmark) -> bool {
    run_padded_base_dft_dispatch_only_metal(&benchmark.inner)
}
