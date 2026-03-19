//! Hidden benchmark support helpers.
//!
//! These helpers expose explicit DFT backend selection to Criterion benches
//! without widening the normal WHIR API surface.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::dense::DenseMatrix;

use crate::whir::dft_backend::run_padded_base_dft_explicit_cpu;

/// Explicit backend choice for padded base-field DFT microbenchmarks.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseDftBenchmarkBackend {
    Cpu,
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    Metal,
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

/// Stable backend label for benchmark names.
#[doc(hidden)]
#[must_use]
pub const fn backend_name(backend: BaseDftBenchmarkBackend) -> &'static str {
    match backend {
        BaseDftBenchmarkBackend::Cpu => "cpu",
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        BaseDftBenchmarkBackend::Metal => "metal",
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
    }
}
