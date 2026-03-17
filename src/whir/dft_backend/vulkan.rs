use p3_dft::TwoAdicSubgroupDft;
use p3_field::TwoAdicField;
use p3_matrix::dense::DenseMatrix;

use super::{DftElementKind, GpuDftJob, run_base_dft_cpu};

const LOCAL_SIZE_X: usize = 256;

#[must_use]
pub(super) const fn is_available() -> bool {
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VulkanDispatch {
    local_size_x: usize,
    workgroups_x: usize,
    workgroups_y: usize,
}

impl VulkanDispatch {
    #[must_use]
    const fn from_job(job: GpuDftJob) -> Self {
        let workgroups_x = job.fft_size.div_ceil(LOCAL_SIZE_X);
        let workgroups_y = job.batch_count;
        Self {
            local_size_x: LOCAL_SIZE_X,
            workgroups_x,
            workgroups_y,
        }
    }
}

/// Vulkan backend entrypoint.
///
/// This is intentionally a CPU fallback until the Vulkan compute pipeline
/// (device init, descriptor sets, dispatch, readback) is implemented.
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
    let dispatch = VulkanDispatch::from_job(job);
    debug_assert_eq!(job.element_kind, DftElementKind::BaseField);
    debug_assert_eq!(dispatch.workgroups_y, job.batch_count);
    run_base_dft_cpu(dft, padded)
}

#[cfg(test)]
mod tests {
    use super::{LOCAL_SIZE_X, VulkanDispatch};
    use crate::whir::dft_backend::{DftElementKind, GpuDftJob};
    use crate::whir::dft_layout::DftBatchLayout;

    #[test]
    fn vulkan_dispatch_matches_round_zero_shape() {
        let layout = DftBatchLayout::for_round(20, 4, 4);
        let job = GpuDftJob::from_layout(DftElementKind::ExtensionField, layout);
        let dispatch = VulkanDispatch::from_job(job);
        assert_eq!(dispatch.local_size_x, LOCAL_SIZE_X);
        assert_eq!(dispatch.workgroups_x, (1 << 18) / LOCAL_SIZE_X);
        assert_eq!(dispatch.workgroups_y, 16);
    }
}
