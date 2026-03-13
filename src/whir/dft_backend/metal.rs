use core::mem::size_of;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalHostBufferView {
    input_bytes: usize,
    output_bytes: usize,
    scratch_bytes: usize,
    twiddle_bytes: usize,
}

impl MetalHostBufferView {
    #[must_use]
    const fn from_submission<T>(submission: MetalSubmission) -> Self {
        let element_bytes = size_of::<T>();
        Self {
            input_bytes: submission.buffers.input_elements * element_bytes,
            output_bytes: submission.buffers.output_elements * element_bytes,
            scratch_bytes: submission.buffers.scratch_elements * element_bytes,
            twiddle_bytes: submission.buffers.twiddle_elements * element_bytes,
        }
    }

    #[must_use]
    const fn total_bytes(self) -> usize {
        self.input_bytes + self.output_bytes + self.scratch_bytes + self.twiddle_bytes
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalRuntimeStatus {
    Ready,
    Unavailable,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalPipelineState {
    Compiled,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalPipelineSet {
    base_field_dft: MetalPipelineState,
    extension_field_dft: MetalPipelineState,
}

impl MetalPipelineSet {
    #[must_use]
    const fn unavailable() -> Self {
        Self {
            base_field_dft: MetalPipelineState::Missing,
            extension_field_dft: MetalPipelineState::Missing,
        }
    }

    #[must_use]
    const fn has_kernel(self, kernel: MetalKernel) -> bool {
        match kernel {
            MetalKernel::BaseFieldDft => {
                matches!(self.base_field_dft, MetalPipelineState::Compiled)
            }
            MetalKernel::ExtensionFieldDft => {
                matches!(self.extension_field_dft, MetalPipelineState::Compiled)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalDeviceContext {
    pipelines: MetalPipelineSet,
}

impl MetalDeviceContext {
    #[must_use]
    const fn unavailable() -> Self {
        Self {
            pipelines: MetalPipelineSet::unavailable(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalRuntime {
    status: MetalRuntimeStatus,
    context: MetalDeviceContext,
}

impl MetalRuntime {
    #[must_use]
    const fn unavailable() -> Self {
        Self {
            status: MetalRuntimeStatus::Unavailable,
            context: MetalDeviceContext::unavailable(),
        }
    }

    #[must_use]
    const fn from_context(context: MetalDeviceContext) -> Self {
        let status = if context.pipelines.has_kernel(MetalKernel::BaseFieldDft)
            && context.pipelines.has_kernel(MetalKernel::ExtensionFieldDft)
        {
            MetalRuntimeStatus::Ready
        } else {
            MetalRuntimeStatus::Unavailable
        };
        Self { status, context }
    }

    #[must_use]
    #[cfg(target_os = "macos")]
    fn detect() -> Self {
        Self::detect_with::<SystemMetalApi>()
    }

    #[must_use]
    #[cfg(not(target_os = "macos"))]
    const fn detect() -> Self {
        Self::unavailable()
    }

    #[must_use]
    fn detect_with<Api: MetalApi>() -> Self {
        match Api::discover_context() {
            Some(context) => Self::from_context(context),
            None => Self::unavailable(),
        }
    }

    #[must_use]
    const fn is_available(self) -> bool {
        matches!(self.status, MetalRuntimeStatus::Ready)
    }

    #[must_use]
    const fn can_submit(self, kernel: MetalKernel) -> bool {
        self.is_available() && self.context.pipelines.has_kernel(kernel)
    }
}

trait MetalApi {
    fn discover_context() -> Option<MetalDeviceContext>;
}

struct SystemMetalApi;

#[cfg(target_os = "macos")]
impl MetalApi for SystemMetalApi {
    fn discover_context() -> Option<MetalDeviceContext> {
        // Real device and pipeline discovery will land here.
        None
    }
}

#[cfg(not(target_os = "macos"))]
impl MetalApi for SystemMetalApi {
    fn discover_context() -> Option<MetalDeviceContext> {
        None
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
    let host_buffers = MetalHostBufferView::from_submission::<F>(submission);
    let runtime = MetalRuntime::detect();
    debug_assert_eq!(submission.plan.kernel, MetalKernel::BaseFieldDft);
    debug_assert!(submission.is_valid());
    debug_assert!(host_buffers.input_bytes > 0);
    debug_assert!(host_buffers.total_bytes() >= host_buffers.input_bytes);
    if runtime.can_submit(submission.plan.kernel) {
        return submit_base_to_metal_runtime(dft, padded, submission, host_buffers, runtime);
    }
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
    let host_buffers = MetalHostBufferView::from_submission::<EF>(submission);
    let runtime = MetalRuntime::detect();
    debug_assert_eq!(submission.plan.kernel, MetalKernel::ExtensionFieldDft);
    debug_assert!(submission.is_valid());
    debug_assert!(host_buffers.input_bytes > 0);
    debug_assert!(host_buffers.total_bytes() >= host_buffers.input_bytes);
    if runtime.can_submit(submission.plan.kernel) {
        return submit_ext_to_metal_runtime(dft, padded, submission, host_buffers, runtime);
    }
    // Placeholder for future Metal upload/dispatch/readback sequence.
    run_ext_dft_cpu(dft, padded)
}

#[inline]
fn submit_base_to_metal_runtime<F, Dft>(
    dft: &Dft,
    padded: DenseMatrix<F>,
    submission: MetalSubmission,
    host_buffers: MetalHostBufferView,
    runtime: MetalRuntime,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert!(runtime.can_submit(submission.plan.kernel));
    debug_assert_eq!(submission.plan.kernel, MetalKernel::BaseFieldDft);
    debug_assert!(host_buffers.total_bytes() >= host_buffers.input_bytes);
    // Placeholder for a real Metal submission path.
    run_base_dft_cpu(dft, padded)
}

#[inline]
fn submit_ext_to_metal_runtime<F, EF, Dft>(
    dft: &Dft,
    padded: DenseMatrix<EF>,
    submission: MetalSubmission,
    host_buffers: MetalHostBufferView,
    runtime: MetalRuntime,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert!(runtime.can_submit(submission.plan.kernel));
    debug_assert_eq!(submission.plan.kernel, MetalKernel::ExtensionFieldDft);
    debug_assert!(host_buffers.total_bytes() >= host_buffers.input_bytes);
    // Placeholder for a real Metal submission path.
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
        MetalApi, MetalBufferLayout, MetalDeviceContext, MetalDispatch, MetalExecutionPlan,
        MetalHostBufferView, MetalKernel, MetalPipelineSet, MetalPipelineState, MetalRuntime,
        MetalRuntimeStatus, MetalSubmission, THREADS_PER_THREADGROUP,
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

    #[test]
    fn metal_host_buffer_view_tracks_byte_sizes() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        let submission = MetalSubmission::from_job(job);
        let buffers = MetalHostBufferView::from_submission::<u32>(submission);
        assert_eq!(buffers.input_bytes, 16 * (1 << 21) * size_of::<u32>());
        assert_eq!(buffers.output_bytes, 16 * (1 << 21) * size_of::<u32>());
        assert_eq!(buffers.scratch_bytes, 16 * (1 << 21) * size_of::<u32>());
        assert_eq!(buffers.twiddle_bytes, (1 << 20) * size_of::<u32>());
        assert_eq!(
            buffers.total_bytes(),
            ((16 * (1 << 21) * 3) + (1 << 20)) * size_of::<u32>()
        );
    }

    #[test]
    fn metal_runtime_defaults_to_unavailable_in_stub() {
        let runtime = MetalRuntime::detect();
        assert_eq!(runtime.status, MetalRuntimeStatus::Unavailable);
        assert!(!runtime.is_available());
        assert_eq!(runtime.context, MetalDeviceContext::unavailable());
        assert!(!runtime.can_submit(MetalKernel::BaseFieldDft));
    }

    #[test]
    fn metal_pipeline_set_reports_missing_kernels() {
        let pipelines = MetalPipelineSet::unavailable();
        assert_eq!(pipelines.base_field_dft, MetalPipelineState::Missing);
        assert_eq!(pipelines.extension_field_dft, MetalPipelineState::Missing);
        assert!(!pipelines.has_kernel(MetalKernel::BaseFieldDft));
        assert!(!pipelines.has_kernel(MetalKernel::ExtensionFieldDft));
    }

    #[test]
    fn metal_runtime_is_ready_when_both_kernels_exist() {
        let context = MetalDeviceContext {
            pipelines: MetalPipelineSet {
                base_field_dft: MetalPipelineState::Compiled,
                extension_field_dft: MetalPipelineState::Compiled,
            },
        };
        let runtime = MetalRuntime::from_context(context);
        assert_eq!(runtime.status, MetalRuntimeStatus::Ready);
        assert!(runtime.is_available());
        assert!(runtime.can_submit(MetalKernel::BaseFieldDft));
        assert!(runtime.can_submit(MetalKernel::ExtensionFieldDft));
    }

    #[test]
    fn metal_runtime_stays_unavailable_with_partial_pipeline_set() {
        let context = MetalDeviceContext {
            pipelines: MetalPipelineSet {
                base_field_dft: MetalPipelineState::Compiled,
                extension_field_dft: MetalPipelineState::Missing,
            },
        };
        let runtime = MetalRuntime::from_context(context);
        assert_eq!(runtime.status, MetalRuntimeStatus::Unavailable);
        assert!(!runtime.is_available());
        assert!(!runtime.can_submit(MetalKernel::BaseFieldDft));
        assert!(!runtime.can_submit(MetalKernel::ExtensionFieldDft));
    }

    struct ReadyMetalApi;

    impl MetalApi for ReadyMetalApi {
        fn discover_context() -> Option<MetalDeviceContext> {
            Some(MetalDeviceContext {
                pipelines: MetalPipelineSet {
                    base_field_dft: MetalPipelineState::Compiled,
                    extension_field_dft: MetalPipelineState::Compiled,
                },
            })
        }
    }

    struct MissingMetalApi;

    impl MetalApi for MissingMetalApi {
        fn discover_context() -> Option<MetalDeviceContext> {
            None
        }
    }

    #[test]
    fn metal_runtime_detect_with_ready_api_is_ready() {
        let runtime = MetalRuntime::detect_with::<ReadyMetalApi>();
        assert_eq!(runtime.status, MetalRuntimeStatus::Ready);
        assert!(runtime.is_available());
        assert!(runtime.can_submit(MetalKernel::BaseFieldDft));
        assert!(runtime.can_submit(MetalKernel::ExtensionFieldDft));
    }

    #[test]
    fn metal_runtime_detect_with_missing_api_is_unavailable() {
        let runtime = MetalRuntime::detect_with::<MissingMetalApi>();
        assert_eq!(runtime.status, MetalRuntimeStatus::Unavailable);
        assert!(!runtime.is_available());
        assert!(!runtime.can_submit(MetalKernel::BaseFieldDft));
    }
}
