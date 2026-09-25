// WGSL compute shader for two-qubit gate application.
//
// Mirrors `simq_state::simd::two_qubit::apply_gate_scalar`'s exact
// convention (the CPU reference this kernel is cross-validated against):
//   - Amplitude index bit `q` (weight 2^q) holds qubit q's value.
//   - The 4x4 gate matrix's local basis index is `2*bit(qubit1) +
//     bit(qubit2)` — qubit1 is the more-significant local bit, independent
//     of which of qubit1/qubit2 has the smaller physical index.
// All buffers use `storage` (see single_qubit_gate.wgsl's header comment
// for why, over `uniform`). Requires wgpu::Features::SHADER_F64 (device
// creation fails loudly, per `gpu.rs`, if an adapter doesn't support it —
// no silent f32 downgrade).

struct Complex {
    re: f64,
    im: f64,
};

@group(0) @binding(0)
var<storage, read_write> state: array<Complex>;
@group(0) @binding(1)
var<storage, read> gate: array<f64, 32>; // row-major 4x4, [re,im] pairs: gate[(r*4+c)*2 + 0/1]
@group(0) @binding(2)
var<storage, read> params: array<u32, 2>; // [qubit1, qubit2]

fn g(r: u32, c: u32) -> Complex {
    let base = (r * 4u + c) * 2u;
    return Complex(gate[base], gate[base + 1u]);
}

fn cmul(a: Complex, b: Complex) -> Complex {
    return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn cadd(a: Complex, b: Complex) -> Complex {
    return Complex(a.re + b.re, a.im + b.im);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = arrayLength(&state);
    let num_groups = dim >> 2u;
    let idx = gid.x;
    // See single_qubit_gate.wgsl's header for why this explicit bound
    // (beyond destination-index checks) is required, not optional.
    if idx >= num_groups {
        return;
    }

    let q1 = params[0];
    let q2 = params[1];
    let low = min(q1, q2);
    let high = max(q1, q2);
    let stride_low = 1u << low;
    let stride_high = 1u << high;

    // Recover this invocation's `i00` base index (both `low` and `high`
    // bits clear) from the flat group index `idx`, matching the CPU
    // scalar kernel's three-level stride loop exactly.
    let groups_per_block = stride_high >> 1u;
    let block = idx / groups_per_block;
    let rem = idx % groups_per_block;
    let mid = rem / stride_low;
    let k = rem % stride_low;
    let base = block * (stride_high * 2u);
    let block_base = base + mid * (stride_low * 2u);
    let i00 = block_base + k;

    let i_low_flip = i00 + stride_low;
    let i_high_flip = i00 + stride_high;
    let i11 = i_high_flip + stride_low;

    // Map the physical low/high-bit-flip positions to matrix indices 1
    // (qubit1=0,qubit2=1) and 2 (qubit1=1,qubit2=0).
    var i01: u32;
    var i10: u32;
    if q1 == low {
        i01 = i_high_flip;
        i10 = i_low_flip;
    } else {
        i01 = i_low_flip;
        i10 = i_high_flip;
    }

    let a0 = state[i00];
    let a1 = state[i01];
    let a2 = state[i10];
    let a3 = state[i11];

    let r0 = cadd(cadd(cmul(g(0u, 0u), a0), cmul(g(0u, 1u), a1)), cadd(cmul(g(0u, 2u), a2), cmul(g(0u, 3u), a3)));
    let r1 = cadd(cadd(cmul(g(1u, 0u), a0), cmul(g(1u, 1u), a1)), cadd(cmul(g(1u, 2u), a2), cmul(g(1u, 3u), a3)));
    let r2 = cadd(cadd(cmul(g(2u, 0u), a0), cmul(g(2u, 1u), a1)), cadd(cmul(g(2u, 2u), a2), cmul(g(2u, 3u), a3)));
    let r3 = cadd(cadd(cmul(g(3u, 0u), a0), cmul(g(3u, 1u), a1)), cadd(cmul(g(3u, 2u), a2), cmul(g(3u, 3u), a3)));

    state[i00] = r0;
    state[i01] = r1;
    state[i10] = r2;
    state[i11] = r3;
}
