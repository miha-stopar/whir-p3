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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalBufferLayout {
    input_elements: usize,
    output_elements: usize,
    scratch_elements: usize,
    twiddle_elements: usize,
}

impl MetalBufferLayout {
    #[must_use]
    const fn from_plan(plan: MetalExecutionPlan) -> Self {
        Self {
            input_elements: plan.input_elements,
            output_elements: plan.input_elements,
            scratch_elements: plan.scratch_elements,
            twiddle_elements: plan.twiddle_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalSubmission {
    plan: MetalExecutionPlan,
    buffers: MetalBufferLayout,
}

impl MetalSubmission {
    #[must_use]
    const fn from_job(job: GpuDftJob) -> Self {
        let plan = MetalExecutionPlan::from_job(job);
        let buffers = MetalBufferLayout::from_plan(plan);
        Self { plan, buffers }
    }

    #[must_use]
    const fn is_valid(self) -> bool {
        self.plan.input_elements > 0
            && self.plan.stage_count > 0
            && self.buffers.input_elements == self.plan.input_elements
            && self.buffers.output_elements == self.plan.input_elements
            && self.buffers.scratch_elements == self.plan.scratch_elements
            && self.buffers.twiddle_elements == self.plan.twiddle_count
    }
}

#[inline]
fn execute_base_submission<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
    submission: MetalSubmission,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert_eq!(submission.plan.kernel, MetalKernel::BaseFieldDft);
    debug_assert!(submission.is_valid());
    // Placeholder for future Metal upload/dispatch/readback sequence.
    run_base_dft_cpu(dft, padded)
}

#[inline]
fn execute_ext_submission<F, EF, Dft>(
    dft: &Dft,
    padded: DenseMatrix<EF>,
    submission: MetalSubmission,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert_eq!(submission.plan.kernel, MetalKernel::ExtensionFieldDft);
    debug_assert!(submission.is_valid());
    // Placeholder for future Metal upload/dispatch/readback sequence.
    run_ext_dft_cpu(dft, padded)
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
    let submission = MetalSubmission::from_job(job);
    debug_assert_eq!(job.element_kind, DftElementKind::BaseField);
    debug_assert_eq!(submission.plan.kernel, MetalKernel::BaseFieldDft);
    debug_assert_eq!(
        submission.plan.dispatch.threadgroups_per_grid_y,
        job.batch_count
    );
    debug_assert_eq!(
        submission.plan.stage_count as usize,
        job.fft_size.ilog2() as usize
    );
    debug_assert_eq!(submission.plan.input_elements, job.element_count);
    execute_base_submission(dft, padded, submission)
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
    let submission = MetalSubmission::from_job(job);
    debug_assert_eq!(job.element_kind, DftElementKind::ExtensionField);
    debug_assert_eq!(submission.plan.kernel, MetalKernel::ExtensionFieldDft);
    debug_assert_eq!(
        submission.plan.dispatch.threadgroups_per_grid_y,
        job.batch_count
    );
    debug_assert_eq!(
        submission.plan.stage_count as usize,
        job.fft_size.ilog2() as usize
    );
    debug_assert_eq!(submission.plan.input_elements, job.element_count);
    execute_ext_submission(dft, padded, submission)
}

#[cfg(test)]
mod tests {
    use super::{
        MetalBufferLayout, MetalDispatch, MetalExecutionPlan, MetalKernel, MetalSubmission,
        THREADS_PER_THREADGROUP,
    };
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

    #[test]
    fn metal_submission_uses_ping_pong_buffers() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        let submission = MetalSubmission::from_job(job);
        assert!(submission.is_valid());
        assert_eq!(
            submission.buffers,
            MetalBufferLayout {
                input_elements: 16 * (1 << 21),
                output_elements: 16 * (1 << 21),
                scratch_elements: 16 * (1 << 21),
                twiddle_elements: 1 << 20,
            }
        );
    }
}
