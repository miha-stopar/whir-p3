use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;

use super::{DftElementKind, GpuDftJob, run_base_dft_cpu, run_ext_dft_cpu};

const THREADS_PER_THREADGROUP: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalKernel {
    BaseFieldDft,
    ExtensionFieldDft,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalDispatch {
    threads_per_threadgroup: usize,
    threadgroups_per_grid_x: usize,
    threadgroups_per_grid_y: usize,
}

impl MetalDispatch {
    #[must_use]
    const fn from_job(job: GpuDftJob) -> Self {
        let threadgroups_per_grid_x = job.fft_size.div_ceil(THREADS_PER_THREADGROUP);
        let threadgroups_per_grid_y = job.batch_count;
        Self {
            threads_per_threadgroup: THREADS_PER_THREADGROUP,
            threadgroups_per_grid_x,
            threadgroups_per_grid_y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalExecutionPlan {
    kernel: MetalKernel,
    dispatch: MetalDispatch,
    stage_count: u32,
    twiddle_count: usize,
    input_elements: usize,
    scratch_elements: usize,
}

impl MetalExecutionPlan {
    #[must_use]
    const fn from_job(job: GpuDftJob) -> Self {
        let kernel = match job.element_kind {
            DftElementKind::BaseField => MetalKernel::BaseFieldDft,
            DftElementKind::ExtensionField => MetalKernel::ExtensionFieldDft,
        };
        Self {
            kernel,
            dispatch: MetalDispatch::from_job(job),
            stage_count: job.fft_size.trailing_zeros(),
            twiddle_count: job.fft_size / 2,
            input_elements: job.element_count,
            // The first GPU implementation will use ping-pong buffers, so scratch
            // matches the uploaded matrix size.
            scratch_elements: job.element_count,
        }
    }
}

/// Metal backend entrypoint.
///
/// This is intentionally a CPU fallback until the Metal kernel pipeline
/// (buffer upload, dispatch, readback) is implemented.
#[inline]
pub(super) fn run_base_dft<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
    job: GpuDftJob,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let plan = MetalExecutionPlan::from_job(job);
    debug_assert_eq!(job.element_kind, DftElementKind::BaseField);
    debug_assert_eq!(plan.kernel, MetalKernel::BaseFieldDft);
    debug_assert_eq!(plan.dispatch.threadgroups_per_grid_y, job.batch_count);
    debug_assert_eq!(plan.stage_count as usize, job.fft_size.ilog2() as usize);
    debug_assert_eq!(plan.input_elements, job.element_count);
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
    job: GpuDftJob,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    let plan = MetalExecutionPlan::from_job(job);
    debug_assert_eq!(job.element_kind, DftElementKind::ExtensionField);
    debug_assert_eq!(plan.kernel, MetalKernel::ExtensionFieldDft);
    debug_assert_eq!(plan.dispatch.threadgroups_per_grid_y, job.batch_count);
    debug_assert_eq!(plan.stage_count as usize, job.fft_size.ilog2() as usize);
    debug_assert_eq!(plan.input_elements, job.element_count);
    run_ext_dft_cpu(dft, padded)
}

#[cfg(test)]
mod tests {
    use super::{MetalDispatch, MetalExecutionPlan, MetalKernel, THREADS_PER_THREADGROUP};
    use crate::whir::dft_backend::{DftElementKind, GpuDftJob};
    use crate::whir::dft_layout::DftBatchLayout;

    #[test]
    fn metal_dispatch_matches_commitment_shape() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        let dispatch = MetalDispatch::from_job(job);
        assert_eq!(dispatch.threads_per_threadgroup, THREADS_PER_THREADGROUP);
        assert_eq!(
            dispatch.threadgroups_per_grid_x,
            (1 << 21) / THREADS_PER_THREADGROUP
        );
        assert_eq!(dispatch.threadgroups_per_grid_y, 16);
    }

    #[test]
    fn metal_execution_plan_matches_commitment_shape() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        let plan = MetalExecutionPlan::from_job(job);
        assert_eq!(plan.kernel, MetalKernel::BaseFieldDft);
        assert_eq!(plan.stage_count, 21);
        assert_eq!(plan.twiddle_count, 1 << 20);
        assert_eq!(plan.input_elements, 16 * (1 << 21));
        assert_eq!(plan.scratch_elements, 16 * (1 << 21));
    }
}
