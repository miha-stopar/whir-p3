use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;

use super::{run_base_dft_cpu, run_ext_dft_cpu};
use crate::whir::dft_layout::DftBatchLayout;

/// Metal backend entrypoint.
///
/// This is intentionally a CPU fallback until the Metal kernel pipeline
/// (buffer upload, dispatch, readback) is implemented.
#[inline]
pub(super) fn run_base_dft<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
    _layout: DftBatchLayout,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    run_base_dft_cpu(dft, padded)
}

/// Metal extension-field backend entrypoint.
///
/// Kept separate from base-field path because extension elements have a different
/// memory representation and will likely use different kernel packing.
#[inline]
pub(super) fn run_ext_dft<F, EF, Dft>(
    dft: &Dft,
    padded: DenseMatrix<EF>,
    _layout: DftBatchLayout,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    run_ext_dft_cpu(dft, padded)
}
