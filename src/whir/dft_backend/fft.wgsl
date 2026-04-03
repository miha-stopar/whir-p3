const FIELD_KIND_BABY_BEAR: u32 = 0u;
const FIELD_KIND_KOALA_BEAR: u32 = 1u;
const BABY_BEAR_MONTY_MU: u32 = 0x88000001u;
const KOALA_BEAR_MONTY_MU: u32 = 0x81000001u;

struct KernelParams {
    word0: u32,
    word1: u32,
    word2: u32,
    word3: u32,
    word4: u32,
    word5: u32,
    word6: u32,
    word7: u32,
}

@group(0) @binding(0)
var<storage, read> input_data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<u32>;

@group(0) @binding(2)
var<storage, read> twiddles: array<u32>;

@group(0) @binding(3)
var<storage, read> kernel_params: KernelParams;

var<workgroup> prefix_tile: array<u32, 1024>;
var<workgroup> stage_pair_tile: array<u32, 1024>;

fn add_mod(a: u32, b: u32, modulus: u32) -> u32 {
    let sum = u64(a) + u64(b);
    if (sum >= u64(modulus)) {
        return u32(sum - u64(modulus));
    }
    return u32(sum);
}

fn sub_mod(a: u32, b: u32, modulus: u32) -> u32 {
    if (a >= b) {
        return a - b;
    }
    return modulus - (b - a);
}

fn monty_mu(field_kind: u32) -> u32 {
    if (field_kind == FIELD_KIND_BABY_BEAR) {
        return BABY_BEAR_MONTY_MU;
    }
    if (field_kind == FIELD_KIND_KOALA_BEAR) {
        return KOALA_BEAR_MONTY_MU;
    }
    return 0u;
}

fn monty_reduce(x: u64, modulus: u32, monty_mu_value: u32) -> u32 {
    let t = u32(x) * monty_mu_value;
    let u = u64(t) * u64(modulus);
    let underflow = x < u;
    let x_sub_u = x - u;
    let x_sub_u_hi = u32(x_sub_u >> 32u);
    var corr = 0u;
    if (underflow) {
        corr = modulus;
    }
    return x_sub_u_hi + corr;
}

fn mul_monty(a: u32, b: u32, field_kind: u32, modulus: u32) -> u32 {
    let product = u64(a) * u64(b);
    return monty_reduce(product, modulus, monty_mu(field_kind));
}

@compute @workgroup_size(256)
fn base_field_dft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_gid = gid.x;
    let butterfly_count = u32(arrayLength(&input_data)) / 2u;
    if (flat_gid >= butterfly_count) {
        return;
    }

    let width = kernel_params.word0;
    let half_size = kernel_params.word1;
    let span = kernel_params.word2;
    let field_kind = kernel_params.word3;
    let modulus = kernel_params.word4;
    let twiddle_offset = kernel_params.word5;

    let pair = flat_gid / width;
    let column = flat_gid - pair * width;
    let offset_in_block = pair % half_size;
    let block = pair / half_size;
    let row = block * span + offset_in_block;
    let lo = row * width + column;
    let hi = lo + half_size * width;

    let a = input_data[lo];
    let b = input_data[hi];
    let twiddle = twiddles[twiddle_offset + offset_in_block];
    let product = mul_monty(b, twiddle, field_kind, modulus);

    output_data[lo] = add_mod(a, product, modulus);
    output_data[hi] = sub_mod(a, product, modulus);
}

@compute @workgroup_size(256)
fn base_field_dft_prefix(
    @builtin(local_invocation_index) local_tid: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let width = kernel_params.word0;
    let tile_rows = kernel_params.word1;
    let log_fft_size = kernel_params.word2;
    let field_kind = kernel_params.word3;
    let modulus = kernel_params.word4;
    let pair_count = width * (tile_rows >> 1u);
    let is_active = local_tid < pair_count;
    let global_base_row = workgroup_id.x * tile_rows;

    var row_lo: u32 = 0u;
    var row_hi: u32 = 0u;
    var column: u32 = 0u;
    var tile_lo: u32 = 0u;
    var tile_hi: u32 = 0u;

    if (is_active) {
        let pair = local_tid / width;
        column = local_tid - pair * width;
        row_lo = pair << 1u;
        row_hi = row_lo + 1u;
        let global_row_lo = global_base_row + row_lo;
        let global_row_hi = global_base_row + row_hi;
        let input_row_lo = reverseBits(global_row_lo) >> (32u - log_fft_size);
        let input_row_hi = reverseBits(global_row_hi) >> (32u - log_fft_size);
        tile_lo = row_lo * width + column;
        tile_hi = row_hi * width + column;
        prefix_tile[tile_lo] = input_data[input_row_lo * width + column];
        prefix_tile[tile_hi] = input_data[input_row_hi * width + column];
    }
    workgroupBarrier();

    var twiddle_offset = 0u;
    var stage = 0u;
    loop {
        if ((1u << stage) >= tile_rows) {
            break;
        }

        if (is_active) {
            let pair = local_tid / width;
            let half_size = 1u << stage;
            let span = half_size << 1u;
            let offset_in_block = pair % half_size;
            let block = pair / half_size;
            let row = block * span + offset_in_block;
            let lo = row * width + column;
            let hi = lo + half_size * width;
            let twiddle = twiddles[twiddle_offset + offset_in_block];
            let a = prefix_tile[lo];
            let b = prefix_tile[hi];
            let product = mul_monty(b, twiddle, field_kind, modulus);
            prefix_tile[lo] = add_mod(a, product, modulus);
            prefix_tile[hi] = sub_mod(a, product, modulus);
            twiddle_offset = twiddle_offset + half_size;
        }
        workgroupBarrier();
        stage = stage + 1u;
    }

    if (is_active) {
        output_data[(global_base_row + row_lo) * width + column] = prefix_tile[tile_lo];
        output_data[(global_base_row + row_hi) * width + column] = prefix_tile[tile_hi];
    }
}

@compute @workgroup_size(256)
fn base_field_dft_stage_pair(
    @builtin(local_invocation_index) local_tid: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let width = kernel_params.word0;
    let columns_per_group = kernel_params.word1;
    let chunk_count = kernel_params.word2;
    let half_size = kernel_params.word3;
    let field_kind = kernel_params.word4;
    let modulus = kernel_params.word5;
    let twiddle_offset = kernel_params.word6;

    let block_rows = half_size << 2u;
    let pair_count = columns_per_group * (block_rows >> 1u);
    let is_active = local_tid < pair_count;
    let chunk_index = workgroup_id.x % chunk_count;
    let block_index = workgroup_id.x / chunk_count;
    let global_base_row = block_index * block_rows;

    var local_column: u32 = 0u;
    var local_pair: u32 = 0u;
    var row_lo: u32 = 0u;
    var row_hi: u32 = 0u;
    var tile_lo: u32 = 0u;
    var tile_hi: u32 = 0u;
    var column: u32 = 0u;

    if (is_active) {
        local_pair = local_tid / columns_per_group;
        local_column = local_tid - local_pair * columns_per_group;
        column = chunk_index * columns_per_group + local_column;
        row_lo = local_pair << 1u;
        row_hi = row_lo + 1u;
        tile_lo = row_lo * columns_per_group + local_column;
        tile_hi = row_hi * columns_per_group + local_column;

        if (column < width) {
            stage_pair_tile[tile_lo] = input_data[(global_base_row + row_lo) * width + column];
            stage_pair_tile[tile_hi] = input_data[(global_base_row + row_hi) * width + column];
        } else {
            stage_pair_tile[tile_lo] = 0u;
            stage_pair_tile[tile_hi] = 0u;
        }
    }
    workgroupBarrier();

    if (is_active) {
        let stage0_half_size = half_size;
        let stage0_span = stage0_half_size << 1u;
        let stage0_offset = local_pair % stage0_half_size;
        let stage0_block = local_pair / stage0_half_size;
        let stage0_row = stage0_block * stage0_span + stage0_offset;
        let stage0_lo = stage0_row * columns_per_group + local_column;
        let stage0_hi = stage0_lo + stage0_half_size * columns_per_group;
        let stage0_twiddle = twiddles[twiddle_offset + stage0_offset];
        let stage0_a = stage_pair_tile[stage0_lo];
        let stage0_b = stage_pair_tile[stage0_hi];
        let stage0_product = mul_monty(stage0_b, stage0_twiddle, field_kind, modulus);
        stage_pair_tile[stage0_lo] = add_mod(stage0_a, stage0_product, modulus);
        stage_pair_tile[stage0_hi] = sub_mod(stage0_a, stage0_product, modulus);
    }
    workgroupBarrier();

    if (is_active) {
        let stage1_half_size = half_size << 1u;
        let stage1_lo = local_pair * columns_per_group + local_column;
        let stage1_hi = stage1_lo + stage1_half_size * columns_per_group;
        let stage1_twiddle = twiddles[twiddle_offset + half_size + local_pair];
        let stage1_a = stage_pair_tile[stage1_lo];
        let stage1_b = stage_pair_tile[stage1_hi];
        let stage1_product = mul_monty(stage1_b, stage1_twiddle, field_kind, modulus);
        stage_pair_tile[stage1_lo] = add_mod(stage1_a, stage1_product, modulus);
        stage_pair_tile[stage1_hi] = sub_mod(stage1_a, stage1_product, modulus);
    }
    workgroupBarrier();

    if (is_active && column < width) {
        output_data[(global_base_row + row_lo) * width + column] = stage_pair_tile[tile_lo];
        output_data[(global_base_row + row_hi) * width + column] = stage_pair_tile[tile_hi];
    }
}
