//! DFT batch layout helpers for WHIR commitment and prover rounds.
//!
//! This module centralizes the matrix-shape formulas used before batched DFT calls.
//! Keeping these formulas in one place helps prevent drift between commitment and
//! round code paths and provides a stable hook point for future GPU execution.

/// Shape information for one batched DFT invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DftBatchLayout {
    /// Number of independent FFT streams (post-transpose matrix width).
    pub batch_count: usize,
    /// Height before RS padding (post-transpose matrix height).
    pub base_height: usize,
    /// Height after RS padding, i.e. FFT size per stream.
    pub padded_height: usize,
}

impl DftBatchLayout {
    /// Build layout for commitment DFT:
    /// - `num_variables = n`
    /// - `folding_factor_0 = k0`
    /// - `starting_log_inv_rate = rho`
    ///
    /// Then:
    /// - batch_count = 2^k0
    /// - base_height = 2^(n-k0)
    /// - padded_height = 2^(n+rho-k0)
    pub(crate) const fn for_commitment(
        num_variables: usize,
        folding_factor_0: usize,
        starting_log_inv_rate: usize,
    ) -> Self {
        let batch_count = 1 << folding_factor_0;
        let base_height = 1 << (num_variables - folding_factor_0);
        let padded_height = 1 << (num_variables + starting_log_inv_rate - folding_factor_0);
        Self {
            batch_count,
            base_height,
            padded_height,
        }
    }

    /// Build layout for a prover round DFT:
    /// - current folded polynomial has `num_variables = n_r`
    /// - next folding factor `k_next`
    /// - inverse rate `inv_rate`
    ///
    /// Then:
    /// - batch_count = 2^k_next
    /// - base_height = 2^(n_r-k_next)
    /// - padded_height = inv_rate * 2^(n_r-k_next)
    pub(crate) const fn for_round(
        num_variables: usize,
        folding_factor_next: usize,
        inv_rate: usize,
    ) -> Self {
        let batch_count = 1 << folding_factor_next;
        let base_height = 1 << (num_variables - folding_factor_next);
        let padded_height = inv_rate * base_height;
        Self {
            batch_count,
            base_height,
            padded_height,
        }
    }

    /// Matrix width passed to `RowMajorMatrixView::new` before transpose.
    ///
    /// The pre-transpose width equals post-transpose base height.
    pub(crate) const fn pre_transpose_width(self) -> usize {
        self.base_height
    }
}

#[cfg(test)]
mod tests {
    use super::DftBatchLayout;

    #[test]
    fn commitment_layout_matches_whir_bench_defaults() {
        // benches/whir.rs defaults:
        // n = 24, k0 = 4, starting_log_inv_rate = 1
        let layout = DftBatchLayout::for_commitment(24, 4, 1);
        assert_eq!(layout.batch_count, 16);
        assert_eq!(layout.base_height, 1 << 20);
        assert_eq!(layout.padded_height, 1 << 21);
    }

    #[test]
    fn round_layout_matches_doc_table_round_0() {
        // After initial fold in the default benchmark setup:
        // n_r = 20, k_next = 4, inv_rate = 4
        let layout = DftBatchLayout::for_round(20, 4, 4);
        assert_eq!(layout.batch_count, 16);
        assert_eq!(layout.base_height, 1 << 16);
        assert_eq!(layout.padded_height, 1 << 18);
    }
}
