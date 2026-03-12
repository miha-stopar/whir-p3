//! DFT execution backend hooks.
//!
//! This module is the integration point for Phase 1 GPU DFT work.
//! CPU remains the source of truth. The optional `gpu` feature currently routes
//! through a GPU hook that falls back to CPU until kernels are implemented.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{DenseMatrix, RowMajorMatrixView},
};

use crate::whir::dft_layout::DftBatchLayout;

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
mod metal;
#[cfg(feature = "gpu-vulkan")]
mod vulkan;

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
pub(crate) fn run_base_dft<F, Dft>(dft: &Dft, evals: &[F], layout: DftBatchLayout) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let padded = reshape_transpose_pad(evals, layout);
    debug_assert_eq!(padded.width(), layout.batch_count);
    debug_assert_eq!(padded.height(), layout.padded_height);
    match selected_backend() {
        DftBackend::Cpu => run_base_dft_cpu(dft, padded),
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => metal::run_base_dft(dft, padded, layout),
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => vulkan::run_base_dft(dft, padded, layout),
    }
}

/// Execute batched DFT for extension-field matrices.
#[inline]
pub(crate) fn run_ext_dft<F, EF, Dft>(
    dft: &Dft,
    evals: &[EF],
    layout: DftBatchLayout,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let padded = reshape_transpose_pad(evals, layout);
    debug_assert_eq!(padded.width(), layout.batch_count);
    debug_assert_eq!(padded.height(), layout.padded_height);
    match selected_backend() {
        DftBackend::Cpu => run_ext_dft_cpu(dft, padded),
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => metal::run_ext_dft(dft, padded, layout),
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => vulkan::run_ext_dft(dft, padded, layout),
    }
}

#[inline]
fn reshape_transpose_pad<T: Field>(evals: &[T], layout: DftBatchLayout) -> DenseMatrix<T> {
    debug_assert_eq!(evals.len(), layout.batch_count * layout.base_height);
    let mut matrix = RowMajorMatrixView::new(evals, layout.pre_transpose_width()).transpose();
    matrix.pad_to_height(layout.padded_height, T::ZERO);
    matrix.to_row_major_matrix()
}

#[inline]
pub(super) fn run_base_dft_cpu<F, Dft>(dft: &Dft, padded: DenseMatrix<F>) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    dft.dft_batch(padded).to_row_major_matrix()
}

#[inline]
pub(super) fn run_ext_dft_cpu<F, EF, Dft>(dft: &Dft, padded: DenseMatrix<EF>) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    dft.dft_algebra_batch(padded).to_row_major_matrix()
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
