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
    #[cfg(feature = "gpu")]
    Gpu,
}

const fn selected_backend() -> DftBackend {
    #[cfg(feature = "gpu")]
    {
        DftBackend::Gpu
    }
    #[cfg(not(feature = "gpu"))]
    {
        DftBackend::Cpu
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
        #[cfg(feature = "gpu")]
        DftBackend::Gpu => run_base_dft_gpu(dft, padded),
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
        #[cfg(feature = "gpu")]
        DftBackend::Gpu => run_ext_dft_gpu(dft, padded),
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

#[cfg(feature = "gpu")]
#[inline]
fn run_base_dft_gpu<F, Dft>(dft: &Dft, padded: DenseMatrix<F>) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    // GPU kernels are wired in a follow-up step.
    run_base_dft_cpu(dft, padded)
}

#[cfg(feature = "gpu")]
#[inline]
fn run_ext_dft_gpu<F, EF, Dft>(dft: &Dft, padded: DenseMatrix<EF>) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    // GPU kernels are wired in a follow-up step.
    run_ext_dft_cpu(dft, padded)
}
