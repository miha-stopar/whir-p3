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
    extension::FlatMatrixView,
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

#[inline]
fn selected_backend() -> DftBackend {
    #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
    if metal::is_available() {
        return DftBackend::Metal;
    }

    #[cfg(feature = "gpu-vulkan")]
    if vulkan::is_available() {
        return DftBackend::Vulkan;
    }

    DftBackend::Cpu
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DftElementKind {
    BaseField,
    ExtensionField,
}

/// GPU-facing description of one batched DFT job.
///
/// This is the stable contract between WHIR's layout math and the eventual
/// Metal/Vulkan kernels. It captures exactly how many FFT streams exist and
/// how long each stream is after padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GpuDftJob {
    pub element_kind: DftElementKind,
    pub batch_count: usize,
    pub fft_size: usize,
    pub element_count: usize,
}

impl GpuDftJob {
    #[must_use]
    pub(super) const fn from_layout(element_kind: DftElementKind, layout: DftBatchLayout) -> Self {
        Self {
            element_kind,
            batch_count: layout.batch_count,
            fft_size: layout.padded_height,
            element_count: layout.batch_count * layout.padded_height,
        }
    }

    #[must_use]
    pub(super) const fn is_valid(self) -> bool {
        self.batch_count > 0
            && self.fft_size > 0
            && self.batch_count.is_power_of_two()
            && self.fft_size.is_power_of_two()
    }

    #[must_use]
    pub(super) const fn flattened_to_base(self, extension_degree: usize) -> Self {
        Self {
            element_kind: DftElementKind::BaseField,
            batch_count: self.batch_count * extension_degree,
            fft_size: self.fft_size,
            element_count: self.element_count * extension_degree,
        }
    }
}

/// Returns the currently selected DFT backend.
#[must_use]
#[allow(dead_code)]
pub(crate) fn selected_backend_name() -> &'static str {
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
    let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
    debug_assert!(job.is_valid());

    match selected_backend() {
        DftBackend::Cpu => {
            let padded = reshape_transpose_pad(evals, layout);
            debug_assert_eq!(padded.width(), layout.batch_count);
            debug_assert_eq!(padded.height(), layout.padded_height);
            run_base_dft_cpu(dft, padded)
        }
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => metal::run_base_dft_from_evals(dft, evals, layout),
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => {
            let padded = reshape_transpose_pad(evals, layout);
            debug_assert_eq!(padded.width(), layout.batch_count);
            debug_assert_eq!(padded.height(), layout.padded_height);
            vulkan::run_base_dft(dft, padded, job)
        }
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
    let job = GpuDftJob::from_layout(DftElementKind::ExtensionField, layout);
    let base_job = job.flattened_to_base(EF::DIMENSION);
    debug_assert!(job.is_valid());
    debug_assert!(base_job.is_valid());

    let base_output = match selected_backend() {
        DftBackend::Cpu => {
            let base_padded = reshape_transpose_pad_ext_to_base(evals, layout);
            debug_assert_eq!(base_padded.width(), layout.batch_count * EF::DIMENSION);
            debug_assert_eq!(base_padded.height(), layout.padded_height);
            run_base_dft_cpu(dft, base_padded)
        }
        #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
        DftBackend::Metal => return metal::run_ext_dft_from_evals(dft, evals, layout),
        #[cfg(feature = "gpu-vulkan")]
        DftBackend::Vulkan => {
            let base_padded = reshape_transpose_pad_ext_to_base(evals, layout);
            debug_assert_eq!(base_padded.width(), layout.batch_count * EF::DIMENSION);
            debug_assert_eq!(base_padded.height(), layout.padded_height);
            vulkan::run_base_dft(dft, base_padded, base_job)
        }
    };

    DenseMatrix::new(
        EF::reconstitute_from_base(base_output.values),
        layout.batch_count,
    )
}

#[inline]
fn reshape_transpose_pad<T: Field>(evals: &[T], layout: DftBatchLayout) -> DenseMatrix<T> {
    debug_assert_eq!(evals.len(), layout.batch_count * layout.base_height);
    let mut matrix = RowMajorMatrixView::new(evals, layout.pre_transpose_width()).transpose();
    matrix.pad_to_height(layout.padded_height, T::ZERO);
    matrix.to_row_major_matrix()
}

#[inline]
fn reshape_transpose_pad_ext_to_base<F, EF>(evals: &[EF], layout: DftBatchLayout) -> DenseMatrix<F>
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(evals.len(), layout.batch_count * layout.base_height);
    let matrix = RowMajorMatrixView::new(evals, layout.pre_transpose_width()).transpose();
    let mut flat = FlatMatrixView::<F, EF, _>::new(matrix).to_row_major_matrix();
    debug_assert_eq!(flat.width(), layout.batch_count * EF::DIMENSION);
    flat.pad_to_height(layout.padded_height, F::ZERO);
    flat
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
pub(super) fn run_padded_base_dft_explicit_cpu<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    run_base_dft_cpu(dft, padded)
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[inline]
pub(super) fn run_padded_base_dft_explicit_metal<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let job = GpuDftJob {
        element_kind: DftElementKind::BaseField,
        batch_count: padded.width(),
        fft_size: padded.height(),
        element_count: padded.width() * padded.height(),
    };
    debug_assert!(job.is_valid());
    metal::run_base_dft(dft, padded, job)
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[must_use]
pub(super) fn metal_is_available_for_bench() -> bool {
    metal::is_available()
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
#[derive(Debug)]
pub(super) struct MetalDispatchOnlyBenchmark {
    inner: metal::MetalDispatchOnlyBenchmark,
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
pub(super) fn prepare_padded_base_dft_dispatch_only_metal<F>(
    padded: &DenseMatrix<F>,
) -> Option<MetalDispatchOnlyBenchmark>
where
    F: TwoAdicField,
{
    metal::prepare_dispatch_only_benchmark(padded).map(|inner| MetalDispatchOnlyBenchmark { inner })
}

#[cfg(all(feature = "gpu-metal", target_os = "macos"))]
pub(super) fn run_padded_base_dft_dispatch_only_metal(
    benchmark: &MetalDispatchOnlyBenchmark,
) -> bool {
    metal::run_dispatch_only_benchmark(&benchmark.inner)
}

#[cfg(test)]
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
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, extension::BinomialExtensionField};

    use super::{
        DftElementKind, GpuDftJob, reshape_transpose_pad, reshape_transpose_pad_ext_to_base,
        run_base_dft_cpu, run_ext_dft, run_ext_dft_cpu, selected_backend_name,
    };
    use crate::whir::dft_layout::DftBatchLayout;

    #[test]
    fn selected_backend_has_known_name() {
        let name = selected_backend_name();
        assert!(matches!(name, "cpu" | "metal" | "vulkan"));
    }

    #[test]
    fn gpu_job_matches_commitment_layout() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        assert_eq!(job.batch_count, 16);
        assert_eq!(job.fft_size, 1 << 21);
        assert_eq!(job.element_count, 16 * (1 << 21));
        assert!(job.is_valid());
    }

    #[test]
    fn extension_gpu_job_flattens_to_base_streams() {
        let layout = DftBatchLayout::for_round(20, 4, 4);
        let job = GpuDftJob::from_layout(DftElementKind::ExtensionField, layout);
        let flattened = job.flattened_to_base(4);
        assert_eq!(flattened.element_kind, DftElementKind::BaseField);
        assert_eq!(flattened.batch_count, 64);
        assert_eq!(flattened.fft_size, 1 << 18);
        assert_eq!(flattened.element_count, 64 * (1 << 18));
        assert!(flattened.is_valid());
    }

    #[test]
    fn reshape_transpose_pad_ext_to_base_matches_flatten_after_padding() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let layout = DftBatchLayout::for_round(4, 2, 2);
        let evals = (0_u32..16)
            .map(|i| EF::from_basis_coefficients_fn(|j| F::from_u32(i * 4 + j as u32 + 1)))
            .collect::<Vec<_>>();

        let expected = reshape_transpose_pad(&evals, layout).flatten_to_base();
        let actual = reshape_transpose_pad_ext_to_base::<F, EF>(&evals, layout);

        assert_eq!(actual, expected);
    }

    #[test]
    fn base_dft_matches_cpu_path() {
        type F = BabyBear;

        let layout = DftBatchLayout::for_commitment(4, 2, 1);
        let evals = (1_u32..=16).map(F::from_u32).collect::<Vec<_>>();
        let dft = Radix2DFTSmallBatch::<F>::default();

        let expected = run_base_dft_cpu(&dft, reshape_transpose_pad(&evals, layout));
        let actual = super::run_base_dft(&dft, &evals, layout);

        assert_eq!(actual, expected);
    }

    #[test]
    fn ext_dft_matches_cpu_algebra_path() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let layout = DftBatchLayout::for_round(4, 2, 2);
        let evals = (0_u32..16)
            .map(|i| EF::from_basis_coefficients_fn(|j| F::from_u32(i * 4 + j as u32 + 1)))
            .collect::<Vec<_>>();
        let dft = Radix2DFTSmallBatch::<F>::default();

        let expected = run_ext_dft_cpu(&dft, reshape_transpose_pad(&evals, layout));
        let actual = run_ext_dft(&dft, &evals, layout);

        assert_eq!(actual, expected);
    }
}
