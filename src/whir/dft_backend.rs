//! DFT execution backend hooks.
//!
//! This module is the integration point for Phase 1 GPU DFT work.
//! CPU remains the source of truth. The optional `gpu` feature currently routes
//! through a GPU hook that falls back to CPU until kernels are implemented.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{Matrix, dense::DenseMatrix};

use crate::whir::dft_layout::DftBatchLayout;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DftBackend {
    Cpu,
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    Metal,
    #[cfg(feature = "gpu-vulkan")]
    Vulkan,
}

const fn selected_backend() -> DftBackend {
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    {
        DftBackend::Metal
    }
    #[cfg(all(
        not(all(feature = "gpu-metal", target_os = "macos")),
        feature = "gpu-vulkan"
    ))]
    {
        DftBackend::Vulkan
    }
    #[cfg(not(any(
        all(feature = "gpu-metal", target_os = "macos"),
        feature = "gpu-vulkan"
    )))]
    {
        DftBackend::Cpu
    }
}

/// Returns the currently selected DFT backend.
#[must_use]
#[allow(dead_code)]
pub(crate) const fn selected_backend_name() -> &'static str {
    match selected_backend() {
        DftBackend::Cpu => "cpu",
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => "metal",
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => "vulkan",
    }
}

/// Execute batched DFT for base-field matrices.
#[inline]
pub(crate) fn run_base_dft<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
    layout: DftBatchLayout,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert_eq!(padded.width(), layout.batch_count);
    debug_assert_eq!(padded.height(), layout.padded_height);
    match selected_backend() {
        DftBackend::Cpu => run_base_dft_cpu(dft, padded),
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => run_base_dft_metal(dft, padded),
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => run_base_dft_vulkan(dft, padded),
    }
}

/// Execute batched DFT for extension-field matrices.
#[inline]
pub(crate) fn run_ext_dft<F, EF, Dft>(
    dft: &Dft,
    padded: DenseMatrix<EF>,
    layout: DftBatchLayout,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert_eq!(padded.width(), layout.batch_count);
    debug_assert_eq!(padded.height(), layout.padded_height);
    match selected_backend() {
        DftBackend::Cpu => run_ext_dft_cpu(dft, padded),
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => run_ext_dft_metal(dft, padded),
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => run_ext_dft_vulkan(dft, padded),
    }
}

#[inline]
fn run_base_dft_cpu<F, Dft>(dft: &Dft, padded: DenseMatrix<F>) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    dft.dft_batch(padded).to_row_major_matrix()
}

#[inline]
fn run_ext_dft_cpu<F, EF, Dft>(dft: &Dft, padded: DenseMatrix<EF>) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    dft.dft_algebra_batch(padded).to_row_major_matrix()
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[inline]
fn run_base_dft_metal<F, Dft>(dft: &Dft, padded: DenseMatrix<F>) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    // Metal kernels are wired in a follow-up step.
    run_base_dft_cpu(dft, padded)
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[inline]
fn run_ext_dft_metal<F, EF, Dft>(dft: &Dft, padded: DenseMatrix<EF>) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    // Metal kernels are wired in a follow-up step.
    run_ext_dft_cpu(dft, padded)
}

#[cfg(feature = "gpu-vulkan")]
#[inline]
fn run_base_dft_vulkan<F, Dft>(dft: &Dft, padded: DenseMatrix<F>) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    // Vulkan kernels are wired in a follow-up step.
    run_base_dft_cpu(dft, padded)
}

#[cfg(feature = "gpu-vulkan")]
#[inline]
fn run_ext_dft_vulkan<F, EF, Dft>(dft: &Dft, padded: DenseMatrix<EF>) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    // Vulkan kernels are wired in a follow-up step.
    run_ext_dft_cpu(dft, padded)
}

#[cfg(test)]
mod tests {
    use super::selected_backend_name;

    #[test]
    fn selected_backend_has_known_name() {
        let name = selected_backend_name();
        assert!(matches!(name, "cpu" | "metal" | "vulkan"));
    }
}
