use core::{any::TypeId, ffi::c_void, mem::size_of, slice};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize, NSRange,
};
use p3_baby_bear::BabyBear;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{PrimeField32, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::{Matrix, dense::DenseMatrix};
#[cfg(all(target_os = "macos", test))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "macos")]
use std::{cell::RefCell, sync::OnceLock, thread_local, vec::Vec};

use super::{DftElementKind, GpuDftJob, run_base_dft_cpu};

const THREADS_PER_THREADGROUP: usize = 256;
const MAX_FUSED_PREFIX_TILE_ROWS: usize = 32;
const MAX_THREADGROUP_TILE_ELEMENTS: usize = 256;

#[cfg(target_os = "macos")]
static METAL_RUNTIME: OnceLock<MetalRuntime> = OnceLock::new();

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_EXECUTOR: RefCell<Option<MetalExecutor>> = const { RefCell::new(None) };
}

#[cfg(all(target_os = "macos", test))]
static METAL_GPU_DISPATCHES: AtomicUsize = AtomicUsize::new(0);

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

constant uint FIELD_KIND_BABY_BEAR = 0;
constant uint FIELD_KIND_KOALA_BEAR = 1;
constant uint PSEUDO_MERSENNE_MASK = 0x7fffffff;
constant uint BABY_BEAR_C = 0x07ffffff;
constant uint KOALA_BEAR_C = 0x00ffffff;

struct StageParams {
    uint width;
    uint half_size;
    uint span;
    uint field_kind;
    uint modulus;
};

struct PrefixParams {
    uint width;
    uint tile_rows;
    uint log_fft_size;
    uint field_kind;
    uint modulus;
};

inline uint add_mod(uint a, uint b, uint modulus) {
    ulong sum = (ulong)a + (ulong)b;
    return sum >= (ulong)modulus ? (uint)(sum - (ulong)modulus) : (uint)sum;
}

inline uint sub_mod(uint a, uint b, uint modulus) {
    return a >= b ? a - b : modulus - (b - a);
}

inline ulong reduce_pseudo_mersenne_once(ulong x, uint c) {
    return (x & (ulong)PSEUDO_MERSENNE_MASK) + (x >> 31) * (ulong)c;
}

inline uint reduce_pseudo_mersenne(ulong x, uint modulus, uint c) {
    while ((x >> 31) != 0) {
        x = reduce_pseudo_mersenne_once(x, c);
    }

    uint reduced = (uint)x;
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    return reduced;
}

inline uint mul_mod(uint a, uint b, uint field_kind, uint modulus) {
    ulong product = (ulong)a * (ulong)b;
    if (field_kind == FIELD_KIND_BABY_BEAR) {
        return reduce_pseudo_mersenne(product, modulus, BABY_BEAR_C);
    }
    if (field_kind == FIELD_KIND_KOALA_BEAR) {
        return reduce_pseudo_mersenne(product, modulus, KOALA_BEAR_C);
    }
    return (uint)(product % (ulong)modulus);
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
    uint product = mul_mod(b, twiddle, params.field_kind, params.modulus);

    output[lo] = add_mod(a, product, params.modulus);
    output[hi] = sub_mod(a, product, params.modulus);
}

kernel void base_field_dft_prefix(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const uint* twiddles [[buffer(2)]],
    constant PrefixParams& params [[buffer(3)]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint tile_index [[threadgroup_position_in_grid]]
) {
    threadgroup uint tile[256];

    uint pair_count = params.width * (params.tile_rows >> 1);
    if (local_tid >= pair_count) {
        return;
    }

    uint pair = local_tid / params.width;
    uint column = local_tid - pair * params.width;
    uint row_lo = pair << 1;
    uint row_hi = row_lo + 1;
    uint global_base_row = tile_index * params.tile_rows;
    uint global_row_lo = global_base_row + row_lo;
    uint global_row_hi = global_base_row + row_hi;
    uint input_row_lo = reverse_bits(global_row_lo) >> (32 - params.log_fft_size);
    uint input_row_hi = reverse_bits(global_row_hi) >> (32 - params.log_fft_size);
    uint tile_lo = row_lo * params.width + column;
    uint tile_hi = row_hi * params.width + column;

    tile[tile_lo] = input[input_row_lo * params.width + column];
    tile[tile_hi] = input[input_row_hi * params.width + column];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint twiddle_offset = 0;
    for (uint stage = 0; (1u << stage) < params.tile_rows; stage++) {
        uint half_size = 1u << stage;
        uint span = half_size << 1;
        uint offset_in_block = pair % half_size;
        uint block = pair / half_size;
        uint row = block * span + offset_in_block;
        uint lo = row * params.width + column;
        uint hi = lo + half_size * params.width;
        uint twiddle = twiddles[twiddle_offset + offset_in_block];
        uint a = tile[lo];
        uint b = tile[hi];
        uint product = mul_mod(b, twiddle, params.field_kind, params.modulus);

        tile[lo] = add_mod(a, product, params.modulus);
        tile[hi] = sub_mod(a, product, params.modulus);
        twiddle_offset += half_size;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    output[(global_base_row + row_lo) * params.width + column] = tile[tile_lo];
    output[(global_base_row + row_hi) * params.width + column] = tile[tile_hi];
}
"#;

#[cfg(target_os = "macos")]
const METAL_BASE_FIELD_KERNEL_NAME: &str = "base_field_dft_stage";
#[cfg(target_os = "macos")]
const METAL_BASE_FIELD_PREFIX_KERNEL_NAME: &str = "base_field_dft_prefix";

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct MetalStageParams {
    width: u32,
    half_size: u32,
    span: u32,
    field_kind: u32,
    modulus: u32,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct MetalPrefixParams {
    width: u32,
    tile_rows: u32,
    log_fft_size: u32,
    field_kind: u32,
    modulus: u32,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MetalPrimeFieldExecution {
    kind: MetalPrimeFieldKind,
    width: usize,
    fft_size: usize,
    modulus: u32,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
struct MetalExecutor {
    device: Device,
    queue: CommandQueue,
    stage_pipeline: ComputePipelineState,
    prefix_pipeline: ComputePipelineState,
    threads_per_threadgroup: usize,
    resources: MetalResourceCache,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalPrimeFieldKind {
    BabyBear,
    KoalaBear,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalTwiddleKey {
    kind: MetalPrimeFieldKind,
    fft_size: usize,
    modulus: u32,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
struct MetalTwiddleCacheEntry {
    key: MetalTwiddleKey,
    buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Default)]
struct MetalResourceCache {
    source: Option<Buffer>,
    work_a: Option<Buffer>,
    work_b: Option<Buffer>,
    twiddles: Vec<MetalTwiddleCacheEntry>,
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
pub(super) struct MetalDispatchOnlyBenchmark {
    execution: MetalPrimeFieldExecution,
    source_buffer: Buffer,
    work_a_buffer: Buffer,
    work_b_buffer: Buffer,
    twiddle_buffer: Buffer,
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

#[cfg(target_os = "macos")]
impl MetalPrimeFieldKind {
    #[must_use]
    const fn shader_field_kind(self) -> u32 {
        match self {
            Self::BabyBear => 0,
            Self::KoalaBear => 1,
        }
    }
}

#[cfg(target_os = "macos")]
impl MetalPrimeFieldExecution {
    #[must_use]
    const fn input_len(&self) -> usize {
        self.width * self.fft_size
    }

    #[must_use]
    const fn field_kind_u32(&self) -> u32 {
        self.kind.shader_field_kind()
    }

    #[must_use]
    fn prefix_tile_rows(&self) -> usize {
        let max_rows_from_width = MAX_THREADGROUP_TILE_ELEMENTS / self.width;
        let max_rows = self
            .fft_size
            .min(MAX_FUSED_PREFIX_TILE_ROWS)
            .min(max_rows_from_width);
        if max_rows < 2 {
            0
        } else {
            1usize << max_rows.ilog2()
        }
    }

    #[must_use]
    fn prefix_stage_count(&self) -> usize {
        let tile_rows = self.prefix_tile_rows();
        if tile_rows < 2 {
            0
        } else {
            tile_rows.ilog2() as usize
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
            twiddle_count: job.fft_size.saturating_sub(1),
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
            twiddle_bytes: submission.buffers.twiddle_elements * size_of::<u32>(),
        }
    }

    #[must_use]
    const fn total_bytes(self) -> usize {
        self.input_bytes + self.output_bytes + self.scratch_bytes + self.twiddle_bytes
    }
}

#[cfg(target_os = "macos")]
impl MetalResourceCache {
    fn ensure_reusable_buffer(
        device: &Device,
        slot: &mut Option<Buffer>,
        min_bytes: u64,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalDiscoveryError> {
        let needs_allocation = slot
            .as_ref()
            .is_none_or(|buffer| buffer.length() < min_bytes);
        if needs_allocation {
            *slot = Some(device.new_buffer(min_bytes, options));
        }

        let Some(buffer) = slot.as_ref() else {
            return Err(MetalDiscoveryError::DeviceUnavailable);
        };
        if buffer.length() < min_bytes {
            return Err(MetalDiscoveryError::DeviceUnavailable);
        }

        Ok(buffer.clone())
    }

    fn execution_buffers(
        &mut self,
        device: &Device,
        input_bytes: u64,
        options: MTLResourceOptions,
    ) -> Result<(Buffer, Buffer, Buffer), MetalDiscoveryError> {
        let source = Self::ensure_reusable_buffer(device, &mut self.source, input_bytes, options)?;
        let work_a = Self::ensure_reusable_buffer(device, &mut self.work_a, input_bytes, options)?;
        let work_b = Self::ensure_reusable_buffer(device, &mut self.work_b, input_bytes, options)?;
        Ok((source, work_a, work_b))
    }

    fn twiddle_buffer(
        &mut self,
        device: &Device,
        key: MetalTwiddleKey,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalDiscoveryError> {
        if let Some(entry) = self.twiddles.iter().find(|entry| entry.key == key) {
            return Ok(entry.buffer.clone());
        }

        let twiddles = stage_twiddles_for_kind(key.kind, key.fft_size);
        let twiddle_len_bytes = (twiddles.len() * size_of::<u32>()) as u64;
        let buffer = device.new_buffer_with_data(
            twiddles.as_ptr().cast::<c_void>(),
            twiddle_len_bytes,
            options,
        );
        if buffer.length() != twiddle_len_bytes {
            return Err(MetalDiscoveryError::DeviceUnavailable);
        }

        self.twiddles.push(MetalTwiddleCacheEntry {
            key,
            buffer: buffer.clone(),
        });
        Ok(buffer)
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
        let stage_function = library
            .get_function(METAL_BASE_FIELD_KERNEL_NAME, None)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let stage_pipeline = device
            .new_compute_pipeline_state_with_function(&stage_function)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let prefix_function = library
            .get_function(METAL_BASE_FIELD_PREFIX_KERNEL_NAME, None)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let prefix_pipeline = device
            .new_compute_pipeline_state_with_function(&prefix_function)
            .map_err(|_| MetalDiscoveryError::MissingPipelines)?;
        let threads_per_threadgroup = stage_pipeline
            .thread_execution_width()
            .min(stage_pipeline.max_total_threads_per_threadgroup())
            .min(THREADS_PER_THREADGROUP as u64) as usize;

        Ok(Self {
            device,
            queue,
            stage_pipeline,
            prefix_pipeline,
            threads_per_threadgroup,
            resources: MetalResourceCache::default(),
        })
    }
}

#[cfg(target_os = "macos")]
fn with_metal_executor<R>(
    f: impl FnOnce(&mut MetalExecutor) -> Result<R, MetalDiscoveryError>,
) -> Result<R, MetalDiscoveryError> {
    METAL_EXECUTOR.with(|slot| {
        if slot.borrow().is_none() {
            let executor = MetalExecutor::new()?;
            *slot.borrow_mut() = Some(executor);
        }

        let mut guard = slot.borrow_mut();
        let executor = guard.as_mut().expect("executor initialized");
        f(executor)
    })
}

#[cfg(target_os = "macos")]
fn dispatch_prime_field_fft(
    executor: &MetalExecutor,
    execution: &MetalPrimeFieldExecution,
    source_buffer: &Buffer,
    work_a_buffer: &Buffer,
    work_b_buffer: &Buffer,
    twiddle_buffer: &Buffer,
) -> Buffer {
    let command_buffer = executor.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let prefix_stage_count = execution.prefix_stage_count();
    let mut twiddle_offset = 0usize;
    let mut src_buffer = source_buffer.clone();
    let mut dst_buffer = work_a_buffer.clone();

    if prefix_stage_count > 0 {
        let tile_rows = execution.prefix_tile_rows();
        let pair_count = execution.width * (tile_rows / 2);
        let tile_count = execution.fft_size / tile_rows;
        let prefix_params = MetalPrefixParams {
            width: execution.width as u32,
            tile_rows: tile_rows as u32,
            log_fft_size: execution.fft_size.ilog2(),
            field_kind: execution.field_kind_u32(),
            modulus: execution.modulus,
        };

        encoder.set_compute_pipeline_state(&executor.prefix_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_buffer(2, Some(twiddle_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<MetalPrefixParams>() as u64,
            (&prefix_params as *const MetalPrefixParams).cast::<c_void>(),
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: tile_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: pair_count as u64,
                height: 1,
                depth: 1,
            },
        );

        twiddle_offset = tile_rows - 1;
        src_buffer = dst_buffer;
        dst_buffer = work_b_buffer.clone();
    }

    encoder.set_compute_pipeline_state(&executor.stage_pipeline);
    let butterfly_count = execution.width * (execution.fft_size / 2);
    let threads_per_threadgroup = executor.threads_per_threadgroup.min(butterfly_count.max(1));

    for stage in prefix_stage_count..execution.fft_size.ilog2() as usize {
        let half_size = 1usize << stage;
        let stage_params = MetalStageParams {
            width: execution.width as u32,
            half_size: half_size as u32,
            span: (half_size << 1) as u32,
            field_kind: execution.field_kind_u32(),
            modulus: execution.modulus,
        };

        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_buffer(
            2,
            Some(twiddle_buffer),
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
    #[cfg(test)]
    METAL_GPU_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    src_buffer
}

#[cfg(target_os = "macos")]
fn execute_prime_field_fft(
    executor: &mut MetalExecutor,
    execution: &MetalPrimeFieldExecution,
    write_input: impl FnOnce(&Buffer),
) -> Result<Buffer, MetalDiscoveryError> {
    let input_len = execution.input_len();
    let buffer_len_bytes = (input_len * size_of::<u32>()) as u64;
    let options = MTLResourceOptions::StorageModeShared;
    let (source_buffer, work_a_buffer, work_b_buffer) =
        executor
            .resources
            .execution_buffers(&executor.device, buffer_len_bytes, options)?;
    let twiddle_key = MetalTwiddleKey {
        kind: execution.kind,
        fft_size: execution.fft_size,
        modulus: execution.modulus,
    };
    let twiddle_buffer =
        executor
            .resources
            .twiddle_buffer(&executor.device, twiddle_key, options)?;

    write_input(&source_buffer);

    debug_assert_eq!(input_len, execution.width * execution.fft_size);
    Ok(dispatch_prime_field_fft(
        executor,
        execution,
        &source_buffer,
        &work_a_buffer,
        &work_b_buffer,
        &twiddle_buffer,
    ))
}

#[cfg(target_os = "macos")]
fn try_prepare_prime_field_execution<F>(padded: &DenseMatrix<F>) -> Option<MetalPrimeFieldExecution>
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        let _ = unsafe { cast_field_slice::<F, BabyBear>(&padded.values) };
        return Some(build_prime_field_execution::<BabyBear>(
            MetalPrimeFieldKind::BabyBear,
            padded.width(),
            padded.height(),
        ));
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        let _ = unsafe { cast_field_slice::<F, KoalaBear>(&padded.values) };
        return Some(build_prime_field_execution::<KoalaBear>(
            MetalPrimeFieldKind::KoalaBear,
            padded.width(),
            padded.height(),
        ));
    }

    None
}

#[cfg(target_os = "macos")]
fn build_prime_field_execution<F>(
    kind: MetalPrimeFieldKind,
    width: usize,
    fft_size: usize,
) -> MetalPrimeFieldExecution
where
    F: PrimeField32 + TwoAdicField,
{
    MetalPrimeFieldExecution {
        kind,
        width,
        fft_size,
        modulus: F::ORDER_U32,
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
fn stage_twiddles_for_kind(kind: MetalPrimeFieldKind, fft_size: usize) -> Vec<u32> {
    match kind {
        MetalPrimeFieldKind::BabyBear => stage_twiddles::<BabyBear>(fft_size),
        MetalPrimeFieldKind::KoalaBear => stage_twiddles::<KoalaBear>(fft_size),
    }
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

#[cfg(target_os = "macos")]
#[inline]
const fn bit_reverse_index(index: usize, log_n: u32) -> usize {
    index.reverse_bits() >> (usize::BITS - log_n)
}

#[cfg(target_os = "macos")]
unsafe fn write_prime_field_matrix_to_buffer<F>(
    values: &[F],
    width: usize,
    fft_size: usize,
    buffer: &Buffer,
) where
    F: PrimeField32,
{
    let encoded = buffer.contents().cast::<u32>();
    let log_h = fft_size.ilog2();

    for dst_row in 0..fft_size {
        let src_row = bit_reverse_index(dst_row, log_h);
        let dst_start = dst_row * width;
        let src_start = src_row * width;
        for column in 0..width {
            // SAFETY: caller provides a buffer with room for `width * fft_size` u32 values.
            unsafe {
                encoded
                    .add(dst_start + column)
                    .write(values[src_start + column].as_canonical_u32())
            };
        }
    }
}

#[cfg(target_os = "macos")]
unsafe fn write_prime_field_matrix_to_buffer_natural<F>(values: &[F], buffer: &Buffer)
where
    F: PrimeField32,
{
    let encoded = buffer.contents().cast::<u32>();
    for (idx, value) in values.iter().enumerate() {
        // SAFETY: caller provides a buffer with room for `values.len()` u32 values.
        unsafe { encoded.add(idx).write(value.as_canonical_u32()) };
    }
}

#[cfg(target_os = "macos")]
fn write_supported_prime_field_input<F>(padded: &DenseMatrix<F>, buffer: &Buffer) -> bool
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        let values = unsafe { cast_field_slice::<F, BabyBear>(&padded.values) };
        unsafe {
            write_prime_field_matrix_to_buffer(values, padded.width(), padded.height(), buffer)
        };
        return true;
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        let values = unsafe { cast_field_slice::<F, KoalaBear>(&padded.values) };
        unsafe {
            write_prime_field_matrix_to_buffer(values, padded.width(), padded.height(), buffer)
        };
        return true;
    }

    false
}

#[cfg(target_os = "macos")]
fn write_supported_prime_field_input_natural<F>(padded: &DenseMatrix<F>, buffer: &Buffer) -> bool
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        let values = unsafe { cast_field_slice::<F, BabyBear>(&padded.values) };
        unsafe { write_prime_field_matrix_to_buffer_natural(values, buffer) };
        return true;
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        let values = unsafe { cast_field_slice::<F, KoalaBear>(&padded.values) };
        unsafe { write_prime_field_matrix_to_buffer_natural(values, buffer) };
        return true;
    }

    false
}

#[cfg(target_os = "macos")]
fn upload_supported_prime_field_input<F>(
    padded: &DenseMatrix<F>,
    execution: &MetalPrimeFieldExecution,
    buffer: &Buffer,
) -> bool
where
    F: TwoAdicField,
{
    let wrote_input = if execution.prefix_stage_count() > 0 {
        write_supported_prime_field_input_natural(padded, buffer)
    } else {
        write_supported_prime_field_input(padded, buffer)
    };
    if wrote_input {
        buffer.did_modify_range(NSRange::new(
            0,
            (execution.input_len() * size_of::<u32>()) as u64,
        ));
    }
    wrote_input
}

#[cfg(all(target_os = "macos", test))]
#[must_use]
fn gpu_dispatch_count() -> usize {
    METAL_GPU_DISPATCHES.load(Ordering::Relaxed)
}

#[cfg(target_os = "macos")]
fn decode_prime_field_output<F>(buffer: &Buffer, len: usize) -> Vec<F>
where
    F: TwoAdicField,
{
    let output_ptr = buffer.contents().cast::<u32>();
    let output = unsafe { slice::from_raw_parts(output_ptr, len) };
    output.iter().copied().map(F::from_u32).collect()
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
        execute_prime_field_fft(executor, &execution, |input_buffer| {
            let wrote_input = upload_supported_prime_field_input(&padded, &execution, input_buffer);
            debug_assert!(
                wrote_input,
                "supported field required for Metal input encoding"
            );
        })
    }) {
        Ok(output_buffer) => DenseMatrix::new(
            decode_prime_field_output(&output_buffer, execution.width * execution.fft_size),
            execution.width,
        ),
        Err(_) => run_base_dft_cpu(dft, padded),
    }
}

#[cfg(target_os = "macos")]
pub(super) fn prepare_dispatch_only_benchmark<F>(
    padded: &DenseMatrix<F>,
) -> Option<MetalDispatchOnlyBenchmark>
where
    F: TwoAdicField,
{
    let execution = try_prepare_prime_field_execution(padded)?;

    with_metal_executor(|executor| {
        let input_bytes = (execution.input_len() * size_of::<u32>()) as u64;
        let options = MTLResourceOptions::StorageModeShared;
        let (source_buffer, work_a_buffer, work_b_buffer) =
            executor
                .resources
                .execution_buffers(&executor.device, input_bytes, options)?;
        let twiddle_buffer = executor.resources.twiddle_buffer(
            &executor.device,
            MetalTwiddleKey {
                kind: execution.kind,
                fft_size: execution.fft_size,
                modulus: execution.modulus,
            },
            options,
        )?;

        if !upload_supported_prime_field_input(padded, &execution, &source_buffer) {
            return Err(MetalDiscoveryError::DeviceUnavailable);
        }

        Ok(MetalDispatchOnlyBenchmark {
            execution,
            source_buffer,
            work_a_buffer,
            work_b_buffer,
            twiddle_buffer,
        })
    })
    .ok()
}

#[cfg(target_os = "macos")]
pub(super) fn run_dispatch_only_benchmark(benchmark: &MetalDispatchOnlyBenchmark) -> bool {
    with_metal_executor(|executor| {
        let _ = dispatch_prime_field_fft(
            executor,
            &benchmark.execution,
            &benchmark.source_buffer,
            &benchmark.work_a_buffer,
            &benchmark.work_b_buffer,
            &benchmark.twiddle_buffer,
        );
        Ok(())
    })
    .is_ok()
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
    use p3_field::{PrimeCharacteristicRing, PrimeField32, TwoAdicField};
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;
    use p3_matrix::dense::DenseMatrix;

    use super::{
        MetalApi, MetalBufferLayout, MetalDeviceContext, MetalDiscoveryError, MetalDispatch,
        MetalExecutionPlan, MetalHostBufferView, MetalKernel, MetalPipelineSet, MetalPipelineState,
        MetalRuntime, MetalRuntimeStatus, MetalSubmission, THREADS_PER_THREADGROUP,
    };
    use crate::whir::dft_backend::{DftElementKind, GpuDftJob, run_base_dft_cpu};
    use crate::whir::dft_layout::DftBatchLayout;

    #[cfg(target_os = "macos")]
    fn next_pseudo_random_u32(state: &mut u64) -> u32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (*state >> 32) as u32
    }

    #[cfg(target_os = "macos")]
    fn pseudo_random_matrix<F>(width: usize, fft_size: usize, seed: u64) -> DenseMatrix<F>
    where
        F: PrimeCharacteristicRing + Send + Sync + Clone,
    {
        let mut state = seed;
        let values = (0..(width * fft_size))
            .map(|_| F::from_u32(next_pseudo_random_u32(&mut state)))
            .collect();
        DenseMatrix::new(values, width)
    }

    #[cfg(target_os = "macos")]
    fn assert_prime_field_gpu_matches_cpu<F>(cases: &[(usize, usize)], seed: u64)
    where
        F: PrimeField32 + TwoAdicField + PrimeCharacteristicRing + Clone,
    {
        if !super::is_available() {
            print_metal_runtime_status("randomized parity skipped");
            return;
        }

        let max_fft_size = cases
            .iter()
            .map(|&(_, fft_size)| fft_size)
            .max()
            .unwrap_or(1);
        let dft = Radix2DFTSmallBatch::<F>::new(max_fft_size);

        for (case_idx, &(width, fft_size)) in cases.iter().enumerate() {
            let padded =
                pseudo_random_matrix::<F>(width, fft_size, seed.wrapping_add(case_idx as u64));
            let job = GpuDftJob {
                element_kind: DftElementKind::BaseField,
                batch_count: width,
                fft_size,
                element_count: width * fft_size,
            };

            let expected = run_base_dft_cpu(&dft, padded.clone());
            let dispatches_before = super::gpu_dispatch_count();
            let actual = super::run_base_dft(&dft, padded, job);
            let dispatches_after = super::gpu_dispatch_count();

            assert_eq!(
                actual, expected,
                "GPU mismatch for width={width}, fft_size={fft_size}"
            );
            assert!(
                dispatches_after > dispatches_before,
                "expected a real Metal dispatch for width={width}, fft_size={fft_size}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn print_metal_runtime_status(label: &str) {
        std::println!(
            "[{label}] Metal available: {}, dispatch count: {}",
            super::is_available(),
            super::gpu_dispatch_count()
        );
    }

    #[cfg(target_os = "macos")]
    fn require_metal_runtime() {
        print_metal_runtime_status("strict check");
        assert!(
            super::is_available(),
            "gpu-metal strict tests require a usable Metal device"
        );
    }

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
        assert_eq!(plan.twiddle_count, (1 << 21) - 1);
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
                twiddle_elements: (1 << 21) - 1,
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
        assert_eq!(buffers.twiddle_bytes, ((1 << 21) - 1) * size_of::<u32>());
        assert_eq!(
            buffers.total_bytes(),
            ((16 * (1 << 21) * 3) + ((1 << 21) - 1)) * size_of::<u32>()
        );
    }

    #[test]
    fn stage_twiddles_match_execution_plan_count() {
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        let job = GpuDftJob::from_layout(DftElementKind::BaseField, layout);
        let plan = MetalExecutionPlan::from_job(job);
        let twiddles = super::stage_twiddles::<BabyBear>(job.fft_size);
        assert_eq!(twiddles.len(), plan.twiddle_count);
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
    fn metal_runtime_status_report() {
        print_metal_runtime_status("status report");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_base_field_dft_matches_cpu_for_baby_bear() {
        if !super::is_available() {
            print_metal_runtime_status("BabyBear parity skipped");
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
        let dispatches_before = super::gpu_dispatch_count();
        let actual = super::run_base_dft(&dft, padded, job);
        let dispatches_after = super::gpu_dispatch_count();
        std::println!(
            "[BabyBear parity] Metal available: true, dispatch count before: {dispatches_before}, after: {dispatches_after}"
        );

        assert_eq!(actual, expected);
        assert!(
            dispatches_after > dispatches_before,
            "expected a real Metal dispatch for BabyBear"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_base_field_dft_matches_cpu_for_koala_bear() {
        if !super::is_available() {
            print_metal_runtime_status("KoalaBear parity skipped");
            return;
        }

        let dft = Radix2DFTSmallBatch::<KoalaBear>::default();
        let padded = DenseMatrix::new((1_u32..=16).map(KoalaBear::from_u32).collect(), 2);
        let job = GpuDftJob {
            element_kind: DftElementKind::BaseField,
            batch_count: 2,
            fft_size: 8,
            element_count: 16,
        };

        let expected = run_base_dft_cpu(&dft, padded.clone());
        let dispatches_before = super::gpu_dispatch_count();
        let actual = super::run_base_dft(&dft, padded, job);
        let dispatches_after = super::gpu_dispatch_count();
        std::println!(
            "[KoalaBear parity] Metal available: true, dispatch count before: {dispatches_before}, after: {dispatches_after}"
        );

        assert_eq!(actual, expected);
        assert!(
            dispatches_after > dispatches_before,
            "expected a real Metal dispatch for KoalaBear"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn randomized_baby_bear_gpu_matches_cpu_across_shapes() {
        let cases = &[(1, 8), (2, 16), (4, 32), (8, 64), (4, 256)];
        assert_prime_field_gpu_matches_cpu::<BabyBear>(cases, 0xBABB1E_u64);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn randomized_koala_bear_gpu_matches_cpu_across_shapes() {
        let cases = &[(1, 8), (2, 16), (4, 32), (8, 64), (4, 256)];
        assert_prime_field_gpu_matches_cpu::<KoalaBear>(cases, 0xC0A1ABu64);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires usable Metal device"]
    fn strict_metal_base_field_dft_matches_cpu_for_baby_bear() {
        require_metal_runtime();
        metal_base_field_dft_matches_cpu_for_baby_bear();
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires usable Metal device"]
    fn strict_metal_base_field_dft_matches_cpu_for_koala_bear() {
        require_metal_runtime();
        metal_base_field_dft_matches_cpu_for_koala_bear();
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires usable Metal device"]
    fn strict_randomized_baby_bear_gpu_matches_cpu_across_shapes() {
        require_metal_runtime();
        let cases = &[(1, 8), (2, 16), (4, 32), (8, 64), (4, 256)];
        assert_prime_field_gpu_matches_cpu::<BabyBear>(cases, 0xBABB1E_u64);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires usable Metal device"]
    fn strict_randomized_koala_bear_gpu_matches_cpu_across_shapes() {
        require_metal_runtime();
        let cases = &[(1, 8), (2, 16), (4, 32), (8, 64), (4, 256)];
        assert_prime_field_gpu_matches_cpu::<KoalaBear>(cases, 0xC0A1ABu64);
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
