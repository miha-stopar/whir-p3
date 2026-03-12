//! DFT execution backend hooks.
//!
//! This module is the integration point for Phase 1 GPU DFT work.
//! For now, both entrypoints route to the existing CPU implementation.

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::DenseMatrix,
};

use crate::whir::dft_layout::DftBatchLayout;

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
    dft.dft_batch(padded).to_row_major_matrix()
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
    dft.dft_algebra_batch(padded).to_row_major_matrix()
}
