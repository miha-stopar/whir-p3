use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;

use super::{run_base_dft_cpu, run_ext_dft_cpu};
use crate::whir::dft_layout::DftBatchLayout;

/// Vulkan backend entrypoint.
///
/// This is intentionally a CPU fallback until the Vulkan compute pipeline
/// (device init, descriptor sets, dispatch, readback) is implemented.
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

/// Vulkan extension-field backend entrypoint.
///
/// Separate hook because extension packing and kernel interfaces differ from
/// base-field kernels.
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
