use core::{any::TypeId, ffi::c_void, mem::size_of, slice};

#[cfg(target_os = "macos")]
use metal::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use p3_baby_bear::BabyBear;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{PrimeField32, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::{Matrix, dense::DenseMatrix, util::reverse_matrix_index_bits};
#[cfg(target_os = "macos")]
use std::{cell::RefCell, sync::OnceLock, thread_local, vec::Vec};

use super::{DftElementKind, GpuDftJob, run_base_dft_cpu};

const THREADS_PER_THREADGROUP: usize = 256;

#[cfg(target_os = "macos")]
static METAL_RUNTIME: OnceLock<MetalRuntime> = OnceLock::new();

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_EXECUTOR: RefCell<Option<MetalExecutor>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
const METAL_DISCOVERY_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void base_field_dft_stub(uint gid [[thread_position_in_grid]]) {
    (void)gid;
}

kernel void extension_field_dft_stub(uint gid [[thread_position_in_grid]]) {
    (void)gid;
}
"#;

#[cfg(target_os = "macos")]
const METAL_BASE_FIELD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct StageParams {
    uint width;
    uint half_size;
    uint span;
    uint modulus;
};

inline uint add_mod(uint a, uint b, uint modulus) {
    ulong sum = (ulong)a + (ulong)b;
    return sum >= (ulong)modulus ? (uint)(sum - (ulong)modulus) : (uint)sum;
}

inline uint sub_mod(uint a, uint b, uint modulus) {
    return a >= b ? a - b : modulus - (b - a);
}

inline uint mul_mod(uint a, uint b, uint modulus) {
    return (uint)(((ulong)a * (ulong)b) % (ulong)modulus);
}

kernel void base_field_dft_stage(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const uint* twiddles [[buffer(2)]],
    constant StageParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair = gid / params.width;
    uint column = gid - pair * params.width;
    uint offset_in_block = pair % params.half_size;
    uint block = pair / params.half_size;
    uint row = block * params.span + offset_in_block;
    uint lo = row * params.width + column;
    uint hi = lo + params.half_size * params.width;

    uint a = input[lo];
    uint b = input[hi];
    uint twiddle = twiddles[offset_in_block];
    uint product = mul_mod(b, twiddle, params.modulus);

    output[lo] = add_mod(a, product, params.modulus);
    output[hi] = sub_mod(a, product, params.modulus);
}
"#;

#[cfg(target_os = "macos")]
const METAL_BASE_FIELD_KERNEL_NAME: &str = "base_field_dft_stage";

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct MetalStageParams {
    width: u32,
    half_size: u32,
    span: u32,
    modulus: u32,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MetalPrimeFieldExecution {
    width: usize,
    modulus: u32,
    input: Vec<u32>,
    twiddles: Vec<u32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MetalExecutor {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    threads_per_threadgroup: usize,
}

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

#[cfg(all(target_os = "macos", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalPreparedSubmission {
    input_bytes: u64,
    output_bytes: u64,
    scratch_bytes: u64,
    twiddle_bytes: u64,
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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalDiscoveryError {
    UnsupportedPlatform,
    DeviceUnavailable,
    MissingPipelines,
}

type MetalDiscoveryResult = Result<MetalDeviceContext, MetalDiscoveryError>;

#[must_use]
pub(super) fn is_available() -> bool {
    MetalRuntime::cached().is_available()
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
    #[cfg(test)]
    fn detect() -> Self {
        Self::detect_with::<SystemMetalApi>()
    }

    #[must_use]
    #[cfg(all(not(target_os = "macos"), test))]
    const fn detect() -> Self {
        Self::unavailable()
    }

    #[must_use]
    #[cfg(target_os = "macos")]
    fn cached() -> Self {
        Self::detect_with_cached::<SystemMetalApi>(&METAL_RUNTIME)
    }

    #[must_use]
    #[cfg(not(target_os = "macos"))]
    const fn cached() -> Self {
        Self::unavailable()
    }

    #[must_use]
    #[cfg(target_os = "macos")]
    fn detect_with_cached<Api: MetalApi>(cache: &OnceLock<Self>) -> Self {
        *cache.get_or_init(Self::detect_with::<Api>)
    }

    #[must_use]
    fn detect_with<Api: MetalApi>() -> Self {
        match Api::discover_context() {
            Ok(context) => Self::from_context(context),
            Err(_) => Self::unavailable(),
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
    fn discover_context() -> MetalDiscoveryResult;
}

struct SystemMetalApi;

#[cfg(target_os = "macos")]
impl MetalApi for SystemMetalApi {
    fn discover_context() -> MetalDiscoveryResult {
        let device = Device::system_default().ok_or(MetalDiscoveryError::DeviceUnavailable)?;
        compile_stub_pipeline_set(&device)?;
        Ok(MetalDeviceContext {
            pipelines: MetalPipelineSet {
                base_field_dft: MetalPipelineState::Compiled,
                extension_field_dft: MetalPipelineState::Compiled,
            },
        })
    }
}

#[cfg(not(target_os = "macos"))]
impl MetalApi for SystemMetalApi {
    fn discover_context() -> MetalDiscoveryResult {
        Err(MetalDiscoveryError::UnsupportedPlatform)
    }
}

#[cfg(target_os = "macos")]
fn compile_stub_pipeline_set(device: &Device) -> Result<(), MetalDiscoveryError> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(METAL_DISCOVERY_SHADER, &options)
        .map_err(|_| MetalDiscoveryError::MissingPipelines)?;

    let base_function = library
        .get_function("base_field_dft_stub", None)
        .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
    device
        .new_compute_pipeline_state_with_function(&base_function)
        .map_err(|_| MetalDiscoveryError::MissingPipelines)?;

    let ext_function = library
        .get_function("extension_field_dft_stub", None)
        .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
    device
        .new_compute_pipeline_state_with_function(&ext_function)
        .map_err(|_| MetalDiscoveryError::MissingPipelines)?;

    Ok(())
}

#[cfg(target_os = "macos")]
impl MetalExecutor {
    fn new() -> Result<Self, MetalDiscoveryError> {
        let device = Device::system_default().ok_or(MetalDiscoveryError::DeviceUnavailable)?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(METAL_BASE_FIELD_SHADER, &options)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let function = library
            .get_function(METAL_BASE_FIELD_KERNEL_NAME, None)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let threads_per_threadgroup = pipeline
            .thread_execution_width()
            .min(pipeline.max_total_threads_per_threadgroup())
            .min(THREADS_PER_THREADGROUP as u64) as usize;

        Ok(Self {
            device,
            queue,
            pipeline,
            threads_per_threadgroup,
        })
    }
}

#[cfg(target_os = "macos")]
fn with_metal_executor<R>(
    f: impl FnOnce(&MetalExecutor) -> Result<R, MetalDiscoveryError>,
) -> Result<R, MetalDiscoveryError> {
    METAL_EXECUTOR.with(|slot| {
        if slot.borrow().is_none() {
            let executor = MetalExecutor::new()?;
            *slot.borrow_mut() = Some(executor);
        }

        let guard = slot.borrow();
        let executor = guard.as_ref().expect("executor initialized");
        f(executor)
    })
}

#[cfg(target_os = "macos")]
fn execute_prime_field_fft(
    executor: &MetalExecutor,
    fft_size: usize,
    execution: &MetalPrimeFieldExecution,
) -> Result<Vec<u32>, MetalDiscoveryError> {
    let buffer_len_bytes = (execution.input.len() * size_of::<u32>()) as u64;
    let twiddle_len_bytes = (execution.twiddles.len() * size_of::<u32>()) as u64;
    let options = MTLResourceOptions::StorageModeShared;

    let input_buffer = executor.device.new_buffer_with_data(
        execution.input.as_ptr().cast::<c_void>(),
        buffer_len_bytes,
        options,
    );
    let scratch_buffer = executor.device.new_buffer(buffer_len_bytes, options);
    let twiddle_buffer = executor.device.new_buffer_with_data(
        execution.twiddles.as_ptr().cast::<c_void>(),
        twiddle_len_bytes,
        options,
    );

    if input_buffer.length() != buffer_len_bytes
        || scratch_buffer.length() != buffer_len_bytes
        || twiddle_buffer.length() != twiddle_len_bytes
    {
        return Err(MetalDiscoveryError::DeviceUnavailable);
    }

    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&executor.pipeline);

    let mut twiddle_offset = 0usize;
    let mut src_buffer = &input_buffer;
    let mut dst_buffer = &scratch_buffer;
    let butterfly_count = execution.width * (fft_size / 2);
    let threads_per_threadgroup = executor.threads_per_threadgroup.min(butterfly_count.max(1));

    for stage in 0..fft_size.ilog2() as usize {
        let half_size = 1usize << stage;
        let stage_params = MetalStageParams {
            width: execution.width as u32,
            half_size: half_size as u32,
            span: (half_size << 1) as u32,
            modulus: execution.modulus,
        };

        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        encoder.set_buffer(
            2,
            Some(&twiddle_buffer),
            (twiddle_offset * size_of::<u32>()) as u64,
        );
        encoder.set_bytes(
            3,
            size_of::<MetalStageParams>() as u64,
            (&stage_params as *const MetalStageParams).cast::<c_void>(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: butterfly_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup as u64,
                height: 1,
                depth: 1,
            },
        );

        twiddle_offset += half_size;
        core::mem::swap(&mut src_buffer, &mut dst_buffer);
    }

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output_ptr = src_buffer.contents().cast::<u32>();
    let output = unsafe { slice::from_raw_parts(output_ptr, execution.input.len()) };
    Ok(output.to_vec())
}

#[cfg(target_os = "macos")]
fn try_prepare_prime_field_execution<F>(padded: &DenseMatrix<F>) -> Option<MetalPrimeFieldExecution>
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        let values = unsafe { cast_field_slice::<F, BabyBear>(&padded.values) };
        return Some(build_prime_field_execution::<BabyBear>(
            values,
            padded.width(),
            padded.height(),
        ));
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        let values = unsafe { cast_field_slice::<F, KoalaBear>(&padded.values) };
        return Some(build_prime_field_execution::<KoalaBear>(
            values,
            padded.width(),
            padded.height(),
        ));
    }

    None
}

#[cfg(target_os = "macos")]
fn build_prime_field_execution<F>(
    values: &[F],
    width: usize,
    fft_size: usize,
) -> MetalPrimeFieldExecution
where
    F: PrimeField32 + TwoAdicField,
{
    let mut input = DenseMatrix::new(
        values.iter().map(PrimeField32::as_canonical_u32).collect(),
        width,
    );
    reverse_matrix_index_bits(&mut input);

    MetalPrimeFieldExecution {
        width,
        modulus: F::ORDER_U32,
        input: input.values,
        twiddles: stage_twiddles::<F>(fft_size),
    }
}

#[cfg(target_os = "macos")]
fn stage_twiddles<F>(fft_size: usize) -> Vec<u32>
where
    F: PrimeField32 + TwoAdicField,
{
    let mut twiddles = Vec::with_capacity(fft_size.saturating_sub(1));
    for stage in 0..fft_size.ilog2() as usize {
        let half_size = 1usize << stage;
        twiddles.extend(
            F::two_adic_generator(stage + 1)
                .powers()
                .take(half_size)
                .map(|twiddle| twiddle.as_canonical_u32()),
        );
    }
    twiddles
}

#[cfg(target_os = "macos")]
unsafe fn cast_field_slice<F, T>(values: &[F]) -> &[T]
where
    F: 'static,
    T: 'static,
{
    debug_assert_eq!(TypeId::of::<F>(), TypeId::of::<T>());
    // SAFETY: callers only use this after an exact `TypeId` equality check, so
    // `F` and `T` are the same concrete type and therefore have identical layout.
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len()) }
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
    let runtime = MetalRuntime::cached();
    debug_assert_eq!(submission.plan.kernel, MetalKernel::BaseFieldDft);
    debug_assert!(submission.is_valid());
    debug_assert!(host_buffers.input_bytes > 0);
    debug_assert!(host_buffers.total_bytes() >= host_buffers.input_bytes);
    if runtime.can_submit(submission.plan.kernel) {
        return submit_base_to_metal_runtime(dft, padded, submission, host_buffers, runtime);
    }
    // Keep CPU as the source of truth when the Metal runtime is unavailable.
    run_base_dft_cpu(dft, padded)
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

    let Some(execution) = try_prepare_prime_field_execution(&padded) else {
        return run_base_dft_cpu(dft, padded);
    };

    match with_metal_executor(|executor| {
        execute_prime_field_fft(executor, padded.height(), &execution)
    }) {
        Ok(values) => DenseMatrix::new(
            values.into_iter().map(F::from_u32).collect(),
            execution.width,
        ),
        Err(_) => run_base_dft_cpu(dft, padded),
    }
}

#[cfg(all(target_os = "macos", test))]
fn prepare_metal_submission(
    host_buffers: MetalHostBufferView,
) -> Result<MetalPreparedSubmission, MetalDiscoveryError> {
    let device = Device::system_default().ok_or(MetalDiscoveryError::DeviceUnavailable)?;
    let _queue = device.new_command_queue();
    let options = MTLResourceOptions::StorageModeShared;

    let input = device.new_buffer(host_buffers.input_bytes as u64, options);
    let output = device.new_buffer(host_buffers.output_bytes as u64, options);
    let scratch = device.new_buffer(host_buffers.scratch_bytes as u64, options);
    let twiddles = device.new_buffer(host_buffers.twiddle_bytes as u64, options);

    if input.length() != host_buffers.input_bytes as u64
        || output.length() != host_buffers.output_bytes as u64
        || scratch.length() != host_buffers.scratch_bytes as u64
        || twiddles.length() != host_buffers.twiddle_bytes as u64
    {
        return Err(MetalDiscoveryError::DeviceUnavailable);
    }

    Ok(MetalPreparedSubmission {
        input_bytes: input.length(),
        output_bytes: output.length(),
        scratch_bytes: scratch.length(),
        twiddle_bytes: twiddles.length(),
    })
}

/// Metal backend entrypoint.
///
/// Supported 32-bit base fields run through the staged Metal kernel path; all
/// other cases fall back to the CPU DFT implementation.
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::DenseMatrix;

    use super::{
        MetalApi, MetalBufferLayout, MetalDeviceContext, MetalDiscoveryError, MetalDispatch,
        MetalExecutionPlan, MetalHostBufferView, MetalKernel, MetalPipelineSet, MetalPipelineState,
        MetalRuntime, MetalRuntimeStatus, MetalSubmission, THREADS_PER_THREADGROUP,
    };
    use crate::whir::dft_backend::{DftElementKind, GpuDftJob, run_base_dft_cpu};
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
    fn metal_runtime_detect_reflects_device_availability() {
        let runtime = MetalRuntime::detect();
        match runtime.status {
            MetalRuntimeStatus::Ready => {
                assert!(runtime.is_available());
                assert!(runtime.can_submit(MetalKernel::BaseFieldDft));
                assert!(runtime.can_submit(MetalKernel::ExtensionFieldDft));
            }
            MetalRuntimeStatus::Unavailable => {
                assert!(!runtime.is_available());
                assert_eq!(runtime.context, MetalDeviceContext::unavailable());
                assert!(!runtime.can_submit(MetalKernel::BaseFieldDft));
                assert!(!runtime.can_submit(MetalKernel::ExtensionFieldDft));
            }
        }
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
        fn discover_context() -> super::MetalDiscoveryResult {
            Ok(MetalDeviceContext {
                pipelines: MetalPipelineSet {
                    base_field_dft: MetalPipelineState::Compiled,
                    extension_field_dft: MetalPipelineState::Compiled,
                },
            })
        }
    }

    struct MissingMetalApi;

    impl MetalApi for MissingMetalApi {
        fn discover_context() -> super::MetalDiscoveryResult {
            Err(MetalDiscoveryError::DeviceUnavailable)
        }
    }

    struct MissingPipelinesMetalApi;

    impl MetalApi for MissingPipelinesMetalApi {
        fn discover_context() -> super::MetalDiscoveryResult {
            Err(MetalDiscoveryError::MissingPipelines)
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

    #[test]
    fn metal_runtime_detect_with_missing_pipelines_api_is_unavailable() {
        let runtime = MetalRuntime::detect_with::<MissingPipelinesMetalApi>();
        assert_eq!(runtime.status, MetalRuntimeStatus::Unavailable);
        assert!(!runtime.is_available());
        assert!(!runtime.can_submit(MetalKernel::BaseFieldDft));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_runtime_cached_only_discovers_once() {
        use std::sync::OnceLock;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DETECTIONS: AtomicUsize = AtomicUsize::new(0);

        struct CountingMetalApi;

        impl MetalApi for CountingMetalApi {
            fn discover_context() -> super::MetalDiscoveryResult {
                DETECTIONS.fetch_add(1, Ordering::SeqCst);
                Ok(MetalDeviceContext {
                    pipelines: MetalPipelineSet {
                        base_field_dft: MetalPipelineState::Compiled,
                        extension_field_dft: MetalPipelineState::Compiled,
                    },
                })
            }
        }

        let cache = OnceLock::new();
        DETECTIONS.store(0, Ordering::SeqCst);

        let first = MetalRuntime::detect_with_cached::<CountingMetalApi>(&cache);
        let second = MetalRuntime::detect_with_cached::<CountingMetalApi>(&cache);

        assert_eq!(DETECTIONS.load(Ordering::SeqCst), 1);
        assert_eq!(first.status, MetalRuntimeStatus::Ready);
        assert_eq!(second.status, MetalRuntimeStatus::Ready);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn system_metal_api_discovers_compiled_stub_pipelines() {
        match super::SystemMetalApi::discover_context() {
            Ok(context) => {
                assert_eq!(
                    context.pipelines.base_field_dft,
                    MetalPipelineState::Compiled
                );
                assert_eq!(
                    context.pipelines.extension_field_dft,
                    MetalPipelineState::Compiled
                );
            }
            Err(MetalDiscoveryError::DeviceUnavailable) => {}
            Err(err) => panic!("unexpected Metal discovery error: {err:?}"),
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prepare_metal_submission_allocates_expected_buffer_sizes() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        let submission = MetalSubmission::from_job(job);
        let host_buffers = MetalHostBufferView::from_submission::<u32>(submission);
        match super::prepare_metal_submission(host_buffers) {
            Ok(prepared) => {
                assert_eq!(prepared.input_bytes, host_buffers.input_bytes as u64);
                assert_eq!(prepared.output_bytes, host_buffers.output_bytes as u64);
                assert_eq!(prepared.scratch_bytes, host_buffers.scratch_bytes as u64);
                assert_eq!(prepared.twiddle_bytes, host_buffers.twiddle_bytes as u64);
            }
            Err(MetalDiscoveryError::DeviceUnavailable) => {}
            Err(err) => panic!("unexpected Metal preparation error: {err:?}"),
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_base_field_dft_matches_cpu_for_baby_bear() {
        if !super::is_available() {
            return;
        }

        let dft = Radix2DFTSmallBatch::<BabyBear>::default();
        let padded = DenseMatrix::new((1_u32..=16).map(BabyBear::from_u32).collect(), 2);
        let job = GpuDftJob {
            element_kind: DftElementKind::BaseField,
            batch_count: 2,
            fft_size: 8,
            element_count: 16,
        };

        let expected = run_base_dft_cpu(&dft, padded.clone());
        let actual = super::run_base_dft(&dft, padded, job);

        assert_eq!(actual, expected);
    }

    #[test]
    fn unsupported_fields_skip_prime_field_metal_path() {
        let padded = DenseMatrix::new(
            [
                Goldilocks::ONE,
                Goldilocks::TWO,
                Goldilocks::from_u32(3),
                Goldilocks::from_u32(4),
            ]
            .to_vec(),
            1,
        );

        assert!(super::try_prepare_prime_field_execution(&padded).is_none());
    }
}
