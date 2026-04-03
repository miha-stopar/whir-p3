use alloc::{sync::Arc, vec::Vec};
use core::{
    any::TypeId,
    mem::{align_of, size_of},
    slice,
};

use p3_baby_bear::BabyBear;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use p3_koala_bear::KoalaBear;
use p3_matrix::{Matrix, dense::DenseMatrix};
use std::sync::{Mutex, OnceLock, mpsc};
use wgpu::util::DeviceExt;

use super::{
    DftElementKind, GpuDftJob, reshape_transpose_pad, reshape_transpose_pad_ext_to_base,
    run_base_dft_cpu,
};
use crate::whir::dft_layout::DftBatchLayout;

const THREADS_PER_WORKGROUP: usize = 256;
const MAX_FUSED_PREFIX_TILE_ROWS: usize = 64;
const MAX_WORKGROUP_TILE_ELEMENTS: usize = 1024;
const WGSL_SHADER_SOURCE: &str = include_str!("fft.wgsl");

static WGSL_EXECUTOR: OnceLock<Result<Mutex<WgslExecutor>, WgslError>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WgslError {
    DeviceUnavailable,
    UnsupportedAdapter,
    MapFailed,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct WgslStageParams {
    width: u32,
    half_size: u32,
    span: u32,
    field_kind: u32,
    modulus: u32,
    twiddle_offset: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct WgslPrefixParams {
    width: u32,
    tile_rows: u32,
    log_fft_size: u32,
    field_kind: u32,
    modulus: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct WgslStagePairParams {
    width: u32,
    columns_per_group: u32,
    chunk_count: u32,
    half_size: u32,
    field_kind: u32,
    modulus: u32,
    twiddle_offset: u32,
    _pad1: u32,
}

#[derive(Debug)]
struct WgslPrimeFieldExecution {
    kind: WgslPrimeFieldKind,
    width: usize,
    fft_size: usize,
    modulus: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WgslPrimeFieldKind {
    BabyBear,
    KoalaBear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WgslTwiddleKey {
    kind: WgslPrimeFieldKind,
    fft_size: usize,
    modulus: u32,
}

#[derive(Debug, Clone)]
struct WgslTwiddleCacheEntry {
    key: WgslTwiddleKey,
    buffer: Arc<wgpu::Buffer>,
}

#[derive(Debug, Default)]
struct WgslResourceCache {
    source: Option<Arc<wgpu::Buffer>>,
    work_a: Option<Arc<wgpu::Buffer>>,
    work_b: Option<Arc<wgpu::Buffer>>,
    staging: Option<Arc<wgpu::Buffer>>,
    stage_params: Option<Arc<wgpu::Buffer>>,
    prefix_params: Option<Arc<wgpu::Buffer>>,
    stage_pair_params: Option<Arc<wgpu::Buffer>>,
    twiddles: Vec<WgslTwiddleCacheEntry>,
    host_words: Vec<u32>,
}

#[derive(Debug)]
struct WgslExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    stage_pipeline: wgpu::ComputePipeline,
    prefix_pipeline: wgpu::ComputePipeline,
    stage_pair_pipeline: wgpu::ComputePipeline,
    stage_bind_group_layout: wgpu::BindGroupLayout,
    prefix_bind_group_layout: wgpu::BindGroupLayout,
    stage_pair_bind_group_layout: wgpu::BindGroupLayout,
    resources: WgslResourceCache,
}

#[derive(Debug)]
pub(super) struct WgslDispatchOnlyBenchmark {
    execution: WgslPrimeFieldExecution,
    source_buffer: Arc<wgpu::Buffer>,
    work_a_buffer: Arc<wgpu::Buffer>,
    work_b_buffer: Arc<wgpu::Buffer>,
    twiddle_buffer: Arc<wgpu::Buffer>,
}

impl WgslPrimeFieldKind {
    #[must_use]
    const fn shader_field_kind(self) -> u32 {
        match self {
            Self::BabyBear => 0,
            Self::KoalaBear => 1,
        }
    }
}

impl WgslPrimeFieldExecution {
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
        let max_tile_elements =
            MAX_WORKGROUP_TILE_ELEMENTS.min(THREADS_PER_WORKGROUP.saturating_mul(2).max(2));
        let max_rows_from_width = max_tile_elements / self.width;
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

    #[must_use]
    fn stage_pair_columns_per_group(&self, stage: usize) -> usize {
        let block_rows = 1usize << (stage + 2);
        let max_tile_elements =
            MAX_WORKGROUP_TILE_ELEMENTS.min(THREADS_PER_WORKGROUP.saturating_mul(2).max(2));
        if block_rows == 0 || block_rows > self.fft_size || block_rows > max_tile_elements {
            return 0;
        }

        self.width.min(max_tile_elements / block_rows)
    }
}

impl WgslResourceCache {
    fn ensure_reusable_buffer(
        device: &wgpu::Device,
        slot: &mut Option<Arc<wgpu::Buffer>>,
        min_bytes: u64,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Result<Arc<wgpu::Buffer>, WgslError> {
        let needs_allocation = slot.as_ref().is_none_or(|buffer| buffer.size() < min_bytes);
        if needs_allocation {
            *slot = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: min_bytes,
                usage,
                mapped_at_creation: false,
            })));
        }

        let Some(buffer) = slot.as_ref() else {
            return Err(WgslError::DeviceUnavailable);
        };
        if buffer.size() < min_bytes {
            return Err(WgslError::DeviceUnavailable);
        }

        Ok(Arc::clone(buffer))
    }

    fn execution_buffers(
        &mut self,
        device: &wgpu::Device,
        input_bytes: u64,
    ) -> Result<
        (
            Arc<wgpu::Buffer>,
            Arc<wgpu::Buffer>,
            Arc<wgpu::Buffer>,
            Arc<wgpu::Buffer>,
        ),
        WgslError,
    > {
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let source = Self::ensure_reusable_buffer(
            device,
            &mut self.source,
            input_bytes,
            storage_usage,
            "WGSL DFT Source Buffer",
        )?;
        let work_a = Self::ensure_reusable_buffer(
            device,
            &mut self.work_a,
            input_bytes,
            storage_usage,
            "WGSL DFT Work A Buffer",
        )?;
        let work_b = Self::ensure_reusable_buffer(
            device,
            &mut self.work_b,
            input_bytes,
            storage_usage,
            "WGSL DFT Work B Buffer",
        )?;
        let staging = Self::ensure_reusable_buffer(
            device,
            &mut self.staging,
            input_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "WGSL DFT Readback Buffer",
        )?;
        Ok((source, work_a, work_b, staging))
    }

    fn params_buffer<T>(
        device: &wgpu::Device,
        slot: &mut Option<Arc<wgpu::Buffer>>,
        label: &'static str,
    ) -> Result<Arc<wgpu::Buffer>, WgslError> {
        Self::ensure_reusable_buffer(
            device,
            slot,
            size_of::<T>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label,
        )
    }

    fn stage_params_buffer(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Arc<wgpu::Buffer>, WgslError> {
        Self::params_buffer::<WgslStageParams>(
            device,
            &mut self.stage_params,
            "WGSL DFT Stage Params",
        )
    }

    fn prefix_params_buffer(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Arc<wgpu::Buffer>, WgslError> {
        Self::params_buffer::<WgslPrefixParams>(
            device,
            &mut self.prefix_params,
            "WGSL DFT Prefix Params",
        )
    }

    fn stage_pair_params_buffer(
        &mut self,
        device: &wgpu::Device,
    ) -> Result<Arc<wgpu::Buffer>, WgslError> {
        Self::params_buffer::<WgslStagePairParams>(
            device,
            &mut self.stage_pair_params,
            "WGSL DFT Stage Pair Params",
        )
    }

    fn twiddle_buffer(
        &mut self,
        device: &wgpu::Device,
        key: WgslTwiddleKey,
    ) -> Result<Arc<wgpu::Buffer>, WgslError> {
        if let Some(entry) = self.twiddles.iter().find(|entry| entry.key == key) {
            return Ok(Arc::clone(&entry.buffer));
        }

        let twiddles = stage_twiddles_for_kind(key.kind, key.fft_size);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("WGSL DFT Twiddles"),
            contents: u32_slice_as_bytes(&twiddles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.twiddles.push(WgslTwiddleCacheEntry {
            key,
            buffer: Arc::new(buffer),
        });
        Ok(Arc::clone(
            &self.twiddles.last().expect("entry inserted").buffer,
        ))
    }

    fn host_words_mut(&mut self, len: usize) -> &mut [u32] {
        self.host_words.resize(len, 0);
        self.host_words.as_mut_slice()
    }
}

impl WgslExecutor {
    fn new() -> Result<Self, WgslError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(WgslError::DeviceUnavailable)?;

        if !adapter.features().contains(wgpu::Features::SHADER_INT64) {
            return Err(WgslError::UnsupportedAdapter);
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("WHIR WGSL DFT Device"),
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|_| WgslError::DeviceUnavailable)?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WHIR WGSL DFT Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL_SHADER_SOURCE.into()),
        });

        let stage_bind_group_layout =
            create_bind_group_layout(&device, "WGSL DFT Stage Bind Group Layout");
        let prefix_bind_group_layout =
            create_bind_group_layout(&device, "WGSL DFT Prefix Bind Group Layout");
        let stage_pair_bind_group_layout =
            create_bind_group_layout(&device, "WGSL DFT Stage Pair Bind Group Layout");

        let stage_pipeline = create_compute_pipeline(
            &device,
            &shader,
            &stage_bind_group_layout,
            "base_field_dft_stage",
            "WGSL DFT Stage Pipeline",
        );
        let prefix_pipeline = create_compute_pipeline(
            &device,
            &shader,
            &prefix_bind_group_layout,
            "base_field_dft_prefix",
            "WGSL DFT Prefix Pipeline",
        );
        let stage_pair_pipeline = create_compute_pipeline(
            &device,
            &shader,
            &stage_pair_bind_group_layout,
            "base_field_dft_stage_pair",
            "WGSL DFT Stage Pair Pipeline",
        );

        Ok(Self {
            device,
            queue,
            stage_pipeline,
            prefix_pipeline,
            stage_pair_pipeline,
            stage_bind_group_layout,
            prefix_bind_group_layout,
            stage_pair_bind_group_layout,
            resources: WgslResourceCache::default(),
        })
    }
}

fn create_bind_group_layout(device: &wgpu::Device, label: &'static str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bind_group_layout: &wgpu::BindGroupLayout,
    entry_point: &'static str,
    label: &'static str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        cache: None,
        compilation_options: Default::default(),
    })
}

fn with_wgsl_executor<R>(
    f: impl FnOnce(&mut WgslExecutor) -> Result<R, WgslError>,
) -> Result<R, WgslError> {
    let executor = WGSL_EXECUTOR.get_or_init(|| WgslExecutor::new().map(Mutex::new));
    let executor = executor.as_ref().map_err(|err| *err)?;
    let mut guard = executor.lock().expect("WGSL executor mutex poisoned");
    f(&mut guard)
}

#[must_use]
pub(super) fn is_available() -> bool {
    with_wgsl_executor(|_| Ok(())).is_ok()
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    twiddles: &wgpu::Buffer,
    params: &wgpu::Buffer,
    label: &'static str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: twiddles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

fn dispatch_prime_field_fft(
    executor: &mut WgslExecutor,
    execution: &WgslPrimeFieldExecution,
    source_buffer: &Arc<wgpu::Buffer>,
    work_a_buffer: &Arc<wgpu::Buffer>,
    work_b_buffer: &Arc<wgpu::Buffer>,
    twiddle_buffer: &Arc<wgpu::Buffer>,
    staging_buffer: Option<&Arc<wgpu::Buffer>>,
) -> Result<Option<Vec<u32>>, WgslError> {
    let mut encoder = executor
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("WGSL DFT Encoder"),
        });
    let prefix_stage_count = execution.prefix_stage_count();
    let mut twiddle_offset = 0usize;
    let mut src_buffer = Arc::clone(source_buffer);
    let mut dst_buffer = Arc::clone(work_a_buffer);

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("WGSL DFT Compute Pass"),
            timestamp_writes: None,
        });

        if prefix_stage_count > 0 {
            let tile_rows = execution.prefix_tile_rows();
            let prefix_params = WgslPrefixParams {
                width: execution.width as u32,
                tile_rows: tile_rows as u32,
                log_fft_size: execution.fft_size.ilog2(),
                field_kind: execution.field_kind_u32(),
                modulus: execution.modulus,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            let prefix_params_buffer = executor.resources.prefix_params_buffer(&executor.device)?;
            executor
                .queue
                .write_buffer(&prefix_params_buffer, 0, any_as_u8_slice(&prefix_params));
            let prefix_bind_group = create_bind_group(
                &executor.device,
                &executor.prefix_bind_group_layout,
                &src_buffer,
                &dst_buffer,
                twiddle_buffer,
                &prefix_params_buffer,
                "WGSL DFT Prefix Bind Group",
            );
            compute_pass.set_pipeline(&executor.prefix_pipeline);
            compute_pass.set_bind_group(0, &prefix_bind_group, &[]);
            compute_pass.dispatch_workgroups((execution.fft_size / tile_rows) as u32, 1, 1);

            twiddle_offset = tile_rows - 1;
            src_buffer = dst_buffer;
            dst_buffer = Arc::clone(work_b_buffer);
        }

        let log_fft_size = execution.fft_size.ilog2() as usize;
        let mut stage = prefix_stage_count;
        while stage < log_fft_size {
            let stage_pair_columns_per_group = if stage + 1 < log_fft_size {
                execution.stage_pair_columns_per_group(stage)
            } else {
                0
            };

            if stage_pair_columns_per_group > 0 {
                let half_size = 1usize << stage;
                let block_rows = half_size << 2;
                let block_count = execution.fft_size / block_rows;
                let chunk_count = execution.width.div_ceil(stage_pair_columns_per_group);
                let stage_pair_params = WgslStagePairParams {
                    width: execution.width as u32,
                    columns_per_group: stage_pair_columns_per_group as u32,
                    chunk_count: chunk_count as u32,
                    half_size: half_size as u32,
                    field_kind: execution.field_kind_u32(),
                    modulus: execution.modulus,
                    twiddle_offset: twiddle_offset as u32,
                    _pad1: 0,
                };
                let stage_pair_params_buffer = executor
                    .resources
                    .stage_pair_params_buffer(&executor.device)?;
                executor.queue.write_buffer(
                    &stage_pair_params_buffer,
                    0,
                    any_as_u8_slice(&stage_pair_params),
                );
                let stage_pair_bind_group = create_bind_group(
                    &executor.device,
                    &executor.stage_pair_bind_group_layout,
                    &src_buffer,
                    &dst_buffer,
                    twiddle_buffer,
                    &stage_pair_params_buffer,
                    "WGSL DFT Stage Pair Bind Group",
                );
                compute_pass.set_pipeline(&executor.stage_pair_pipeline);
                compute_pass.set_bind_group(0, &stage_pair_bind_group, &[]);
                compute_pass.dispatch_workgroups((block_count * chunk_count) as u32, 1, 1);

                twiddle_offset += half_size + (half_size << 1);
                stage += 2;
                core::mem::swap(&mut src_buffer, &mut dst_buffer);
                continue;
            }

            let half_size = 1usize << stage;
            let stage_params = WgslStageParams {
                width: execution.width as u32,
                half_size: half_size as u32,
                span: (half_size << 1) as u32,
                field_kind: execution.field_kind_u32(),
                modulus: execution.modulus,
                twiddle_offset: twiddle_offset as u32,
                _pad1: 0,
                _pad2: 0,
            };
            let stage_params_buffer = executor.resources.stage_params_buffer(&executor.device)?;
            executor
                .queue
                .write_buffer(&stage_params_buffer, 0, any_as_u8_slice(&stage_params));
            let stage_bind_group = create_bind_group(
                &executor.device,
                &executor.stage_bind_group_layout,
                &src_buffer,
                &dst_buffer,
                twiddle_buffer,
                &stage_params_buffer,
                "WGSL DFT Stage Bind Group",
            );
            let butterfly_count = execution.width * (execution.fft_size / 2);
            compute_pass.set_pipeline(&executor.stage_pipeline);
            compute_pass.set_bind_group(0, &stage_bind_group, &[]);
            compute_pass.dispatch_workgroups(
                butterfly_count.div_ceil(THREADS_PER_WORKGROUP) as u32,
                1,
                1,
            );

            twiddle_offset += half_size;
            stage += 1;
            core::mem::swap(&mut src_buffer, &mut dst_buffer);
        }
    }

    let output_bytes = (execution.input_len() * size_of::<u32>()) as u64;
    if let Some(staging_buffer) = staging_buffer {
        encoder.copy_buffer_to_buffer(&src_buffer, 0, staging_buffer, 0, output_bytes);
    }
    executor.queue.submit(Some(encoder.finish()));

    if staging_buffer.is_none() {
        let _ = executor.device.poll(wgpu::Maintain::wait());
        return Ok(None);
    }

    let staging_buffer = staging_buffer.expect("staging buffer checked");
    let buffer_slice = staging_buffer.slice(..output_bytes);
    let (tx, rx) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = executor.device.poll(wgpu::Maintain::wait());
    let map_result = rx.recv().map_err(|_| WgslError::MapFailed)?;
    map_result.map_err(|_| WgslError::MapFailed)?;

    let data = buffer_slice.get_mapped_range();
    let (head, words, tail) = unsafe { data.align_to::<u32>() };
    debug_assert!(head.is_empty());
    debug_assert!(tail.is_empty());
    let output = words.to_vec();
    drop(data);
    staging_buffer.unmap();
    Ok(Some(output))
}

fn execute_prime_field_fft(
    executor: &mut WgslExecutor,
    execution: &WgslPrimeFieldExecution,
    fill_input: impl FnOnce(&mut [u32], bool),
) -> Result<Vec<u32>, WgslError> {
    let input_len = execution.input_len();
    let input_bytes = (input_len * size_of::<u32>()) as u64;
    let (source_buffer, work_a_buffer, work_b_buffer, staging_buffer) = executor
        .resources
        .execution_buffers(&executor.device, input_bytes)?;
    let twiddle_buffer = executor.resources.twiddle_buffer(
        &executor.device,
        WgslTwiddleKey {
            kind: execution.kind,
            fft_size: execution.fft_size,
            modulus: execution.modulus,
        },
    )?;

    let use_natural_order = execution.prefix_stage_count() > 0;
    let host_words = executor.resources.host_words_mut(input_len);
    fill_input(host_words, use_natural_order);
    executor
        .queue
        .write_buffer(&source_buffer, 0, u32_slice_as_bytes(host_words));

    dispatch_prime_field_fft(
        executor,
        execution,
        &source_buffer,
        &work_a_buffer,
        &work_b_buffer,
        &twiddle_buffer,
        Some(&staging_buffer),
    )
    .map(|output| output.expect("readback requested"))
}

fn try_prepare_prime_field_execution_for_dims<F>(
    width: usize,
    fft_size: usize,
) -> Option<WgslPrimeFieldExecution>
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        return Some(build_prime_field_execution::<BabyBear>(
            WgslPrimeFieldKind::BabyBear,
            width,
            fft_size,
        ));
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        return Some(build_prime_field_execution::<KoalaBear>(
            WgslPrimeFieldKind::KoalaBear,
            width,
            fft_size,
        ));
    }

    None
}

fn try_prepare_prime_field_execution<F>(padded: &DenseMatrix<F>) -> Option<WgslPrimeFieldExecution>
where
    F: TwoAdicField,
{
    try_prepare_prime_field_execution_for_dims::<F>(padded.width(), padded.height())
}

fn build_prime_field_execution<F>(
    kind: WgslPrimeFieldKind,
    width: usize,
    fft_size: usize,
) -> WgslPrimeFieldExecution
where
    F: PrimeField32 + TwoAdicField,
{
    WgslPrimeFieldExecution {
        kind,
        width,
        fft_size,
        modulus: F::ORDER_U32,
    }
}

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
                .map(|twiddle| twiddle.to_unique_u32()),
        );
    }
    twiddles
}

fn stage_twiddles_for_kind(kind: WgslPrimeFieldKind, fft_size: usize) -> Vec<u32> {
    match kind {
        WgslPrimeFieldKind::BabyBear => stage_twiddles::<BabyBear>(fft_size),
        WgslPrimeFieldKind::KoalaBear => stage_twiddles::<KoalaBear>(fft_size),
    }
}

unsafe fn cast_field_slice<F, T>(values: &[F]) -> &[T]
where
    F: 'static,
    T: 'static,
{
    debug_assert_eq!(TypeId::of::<F>(), TypeId::of::<T>());
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<T>(), values.len()) }
}

#[inline]
const fn bit_reverse_index(index: usize, log_n: u32) -> usize {
    index.reverse_bits() >> (usize::BITS - log_n)
}

#[inline]
const fn logical_row_index(dst_row: usize, padded_height: usize, use_natural_order: bool) -> usize {
    if use_natural_order {
        dst_row
    } else {
        bit_reverse_index(dst_row, padded_height.ilog2())
    }
}

fn serialize_prime_field_evals_to_words<F>(
    values: &[F],
    layout: DftBatchLayout,
    use_natural_order: bool,
    output: &mut [u32],
) where
    F: PrimeField32,
{
    debug_assert_eq!(values.len(), layout.batch_count * layout.base_height);
    debug_assert_eq!(output.len(), layout.batch_count * layout.padded_height);

    output.fill(0);
    for dst_row in 0..layout.padded_height {
        let src_row = logical_row_index(dst_row, layout.padded_height, use_natural_order);
        if src_row >= layout.base_height {
            continue;
        }

        let dst_start = dst_row * layout.batch_count;
        for column in 0..layout.batch_count {
            output[dst_start + column] =
                values[column * layout.base_height + src_row].to_unique_u32();
        }
    }
}

fn serialize_extension_field_evals_to_words_with_repr<F, EF, T>(
    values: &[EF],
    layout: DftBatchLayout,
    use_natural_order: bool,
    output: &mut [u32],
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    T: PrimeField32 + 'static,
{
    let width = layout.batch_count * EF::DIMENSION;
    debug_assert_eq!(values.len(), layout.batch_count * layout.base_height);
    debug_assert_eq!(output.len(), width * layout.padded_height);

    output.fill(0);
    for dst_row in 0..layout.padded_height {
        let src_row = logical_row_index(dst_row, layout.padded_height, use_natural_order);
        if src_row >= layout.base_height {
            continue;
        }

        let dst_start = dst_row * width;
        for column in 0..layout.batch_count {
            let coeffs = unsafe {
                cast_field_slice::<F, T>(
                    values[column * layout.base_height + src_row].as_basis_coefficients_slice(),
                )
            };
            let coeff_start = dst_start + column * EF::DIMENSION;
            for (idx, coeff) in coeffs.iter().enumerate() {
                output[coeff_start + idx] = coeff.to_unique_u32();
            }
        }
    }
}

fn serialize_prime_field_matrix_to_words<F>(
    values: &[F],
    width: usize,
    fft_size: usize,
    use_natural_order: bool,
    output: &mut [u32],
) where
    F: PrimeField32,
{
    debug_assert_eq!(values.len(), width * fft_size);
    debug_assert_eq!(output.len(), width * fft_size);

    for dst_row in 0..fft_size {
        let src_row = logical_row_index(dst_row, fft_size, use_natural_order);
        let dst_start = dst_row * width;
        let src_start = src_row * width;
        for column in 0..width {
            output[dst_start + column] = values[src_start + column].to_unique_u32();
        }
    }
}

fn write_supported_prime_field_evals_input<F>(
    evals: &[F],
    layout: DftBatchLayout,
    use_natural_order: bool,
    output: &mut [u32],
) -> bool
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        let values = unsafe { cast_field_slice::<F, BabyBear>(evals) };
        serialize_prime_field_evals_to_words(values, layout, use_natural_order, output);
        return true;
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        let values = unsafe { cast_field_slice::<F, KoalaBear>(evals) };
        serialize_prime_field_evals_to_words(values, layout, use_natural_order, output);
        return true;
    }

    false
}

fn write_supported_extension_field_evals_input<F, EF>(
    evals: &[EF],
    layout: DftBatchLayout,
    use_natural_order: bool,
    output: &mut [u32],
) -> bool
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        serialize_extension_field_evals_to_words_with_repr::<F, EF, BabyBear>(
            evals,
            layout,
            use_natural_order,
            output,
        );
        return true;
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        serialize_extension_field_evals_to_words_with_repr::<F, EF, KoalaBear>(
            evals,
            layout,
            use_natural_order,
            output,
        );
        return true;
    }

    false
}

fn write_supported_prime_field_input<F>(
    padded: &DenseMatrix<F>,
    use_natural_order: bool,
    output: &mut [u32],
) -> bool
where
    F: TwoAdicField,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>() {
        let values = unsafe { cast_field_slice::<F, BabyBear>(&padded.values) };
        serialize_prime_field_matrix_to_words(
            values,
            padded.width(),
            padded.height(),
            use_natural_order,
            output,
        );
        return true;
    }

    if TypeId::of::<F>() == TypeId::of::<KoalaBear>() {
        let values = unsafe { cast_field_slice::<F, KoalaBear>(&padded.values) };
        serialize_prime_field_matrix_to_words(
            values,
            padded.width(),
            padded.height(),
            use_natural_order,
            output,
        );
        return true;
    }

    false
}

unsafe fn reinterpret_u32_vec_as_field<F>(words: Vec<u32>) -> Vec<F>
where
    F: 'static,
{
    debug_assert_eq!(size_of::<F>(), size_of::<u32>());
    debug_assert_eq!(align_of::<F>(), align_of::<u32>());
    let (ptr, len, cap) = words.into_raw_parts();
    unsafe { Vec::from_raw_parts(ptr.cast::<F>(), len, cap) }
}

fn decode_supported_prime_field_words<F>(words: &[u32]) -> Option<Vec<F>>
where
    F: TwoAdicField + 'static,
{
    if TypeId::of::<F>() == TypeId::of::<BabyBear>()
        || TypeId::of::<F>() == TypeId::of::<KoalaBear>()
    {
        return Some(unsafe { reinterpret_u32_vec_as_field(words.to_vec()) });
    }

    None
}

#[inline]
pub(super) fn run_base_dft_from_evals<F, Dft>(
    dft: &Dft,
    evals: &[F],
    layout: DftBatchLayout,
) -> DenseMatrix<F>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert_eq!(evals.len(), layout.batch_count * layout.base_height);
    let Some(execution) =
        try_prepare_prime_field_execution_for_dims::<F>(layout.batch_count, layout.padded_height)
    else {
        return run_base_dft_cpu(dft, reshape_transpose_pad(evals, layout));
    };

    match with_wgsl_executor(|executor| {
        execute_prime_field_fft(executor, &execution, |output, use_natural_order| {
            let wrote_input =
                write_supported_prime_field_evals_input(evals, layout, use_natural_order, output);
            debug_assert!(
                wrote_input,
                "supported field required for WGSL input encoding"
            );
        })
    }) {
        Ok(words) => {
            let Some(values) = decode_supported_prime_field_words::<F>(&words) else {
                return run_base_dft_cpu(dft, reshape_transpose_pad(evals, layout));
            };
            DenseMatrix::new(values, layout.batch_count)
        }
        Err(_) => run_base_dft_cpu(dft, reshape_transpose_pad(evals, layout)),
    }
}

#[inline]
pub(super) fn run_ext_dft_from_evals<F, EF, Dft>(
    dft: &Dft,
    evals: &[EF],
    layout: DftBatchLayout,
) -> DenseMatrix<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    debug_assert_eq!(evals.len(), layout.batch_count * layout.base_height);
    let flattened_width = layout.batch_count * EF::DIMENSION;
    let Some(execution) =
        try_prepare_prime_field_execution_for_dims::<F>(flattened_width, layout.padded_height)
    else {
        let base_padded = reshape_transpose_pad_ext_to_base::<F, EF>(evals, layout);
        let base_output = run_base_dft_cpu(dft, base_padded);
        return DenseMatrix::new(
            EF::reconstitute_from_base(base_output.values),
            layout.batch_count,
        );
    };

    match with_wgsl_executor(|executor| {
        execute_prime_field_fft(executor, &execution, |output, use_natural_order| {
            let wrote_input = write_supported_extension_field_evals_input::<F, EF>(
                evals,
                layout,
                use_natural_order,
                output,
            );
            debug_assert!(
                wrote_input,
                "supported field required for WGSL extension-field input encoding"
            );
        })
    }) {
        Ok(words) => {
            let Some(values) = decode_supported_prime_field_words::<F>(&words) else {
                let base_padded = reshape_transpose_pad_ext_to_base::<F, EF>(evals, layout);
                let base_output = run_base_dft_cpu(dft, base_padded);
                return DenseMatrix::new(
                    EF::reconstitute_from_base(base_output.values),
                    layout.batch_count,
                );
            };
            DenseMatrix::new(EF::reconstitute_from_base(values), layout.batch_count)
        }
        Err(_) => {
            let base_padded = reshape_transpose_pad_ext_to_base::<F, EF>(evals, layout);
            let base_output = run_base_dft_cpu(dft, base_padded);
            DenseMatrix::new(
                EF::reconstitute_from_base(base_output.values),
                layout.batch_count,
            )
        }
    }
}

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
    debug_assert_eq!(job.element_kind, DftElementKind::BaseField);
    let Some(execution) = try_prepare_prime_field_execution(&padded) else {
        return run_base_dft_cpu(dft, padded);
    };

    match with_wgsl_executor(|executor| {
        execute_prime_field_fft(executor, &execution, |output, use_natural_order| {
            let wrote_input = write_supported_prime_field_input(&padded, use_natural_order, output);
            debug_assert!(
                wrote_input,
                "supported field required for WGSL padded input encoding"
            );
        })
    }) {
        Ok(words) => {
            let Some(values) = decode_supported_prime_field_words::<F>(&words) else {
                return run_base_dft_cpu(dft, padded);
            };
            DenseMatrix::new(values, execution.width)
        }
        Err(_) => run_base_dft_cpu(dft, padded),
    }
}

pub(super) fn prepare_dispatch_only_benchmark<F>(
    padded: &DenseMatrix<F>,
) -> Option<WgslDispatchOnlyBenchmark>
where
    F: TwoAdicField,
{
    let execution = try_prepare_prime_field_execution(padded)?;

    with_wgsl_executor(|executor| {
        let input_bytes = (execution.input_len() * size_of::<u32>()) as u64;
        let (source_buffer, work_a_buffer, work_b_buffer, _staging_buffer) = executor
            .resources
            .execution_buffers(&executor.device, input_bytes)?;
        let twiddle_buffer = executor.resources.twiddle_buffer(
            &executor.device,
            WgslTwiddleKey {
                kind: execution.kind,
                fft_size: execution.fft_size,
                modulus: execution.modulus,
            },
        )?;

        let use_natural_order = execution.prefix_stage_count() > 0;
        let host_words = executor.resources.host_words_mut(execution.input_len());
        if !write_supported_prime_field_input(padded, use_natural_order, host_words) {
            return Err(WgslError::DeviceUnavailable);
        }
        executor
            .queue
            .write_buffer(&source_buffer, 0, u32_slice_as_bytes(host_words));

        Ok(WgslDispatchOnlyBenchmark {
            execution,
            source_buffer,
            work_a_buffer,
            work_b_buffer,
            twiddle_buffer,
        })
    })
    .ok()
}

pub(super) fn run_dispatch_only_benchmark(benchmark: &WgslDispatchOnlyBenchmark) -> bool {
    with_wgsl_executor(|executor| {
        let _ = dispatch_prime_field_fft(
            executor,
            &benchmark.execution,
            &benchmark.source_buffer,
            &benchmark.work_a_buffer,
            &benchmark.work_b_buffer,
            &benchmark.twiddle_buffer,
            None,
        )?;
        Ok(())
    })
    .is_ok()
}

#[inline]
fn any_as_u8_slice<T>(value: &T) -> &[u8] {
    unsafe { slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

#[inline]
fn u32_slice_as_bytes(values: &[u32]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values)) }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;

    use super::{is_available, run_base_dft_from_evals};
    use crate::whir::dft_backend::{reshape_transpose_pad, run_base_dft_cpu};
    use crate::whir::dft_layout::DftBatchLayout;

    #[test]
    fn wgsl_base_field_dft_matches_cpu_for_baby_bear() {
        if !is_available() {
            std::eprintln!("WGSL unavailable; skipping BabyBear parity test");
            return;
        }

        let layout = DftBatchLayout::for_commitment(4, 2, 1);
        let evals = (1_u32..=16).map(BabyBear::from_u32).collect::<Vec<_>>();
        let dft = Radix2DFTSmallBatch::<BabyBear>::default();
        let expected = run_base_dft_cpu(&dft, reshape_transpose_pad(&evals, layout));
        let actual = run_base_dft_from_evals(&dft, &evals, layout);
        assert_eq!(actual, expected);
    }

    #[test]
    fn wgsl_base_field_dft_matches_cpu_for_koala_bear() {
        if !is_available() {
            std::eprintln!("WGSL unavailable; skipping KoalaBear parity test");
            return;
        }

        let layout = DftBatchLayout::for_commitment(4, 2, 1);
        let evals = (1_u32..=16).map(KoalaBear::from_u32).collect::<Vec<_>>();
        let dft = Radix2DFTSmallBatch::<KoalaBear>::default();
        let expected = run_base_dft_cpu(&dft, reshape_transpose_pad(&evals, layout));
        let actual = run_base_dft_from_evals(&dft, &evals, layout);
        assert_eq!(actual, expected);
    }
}
