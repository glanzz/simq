// WGSL compute shader for single-qubit gate application.
// Applies a 2x2 gate matrix to a dense state vector.
//
// All buffers use the `storage` address space rather than `uniform`: WGSL
// requires uniform-address-space array strides to be 16-byte-aligned,
// which a flat `array<f64, N>` does not satisfy without manual padding —
// `storage` buffers only require natural (8-byte, for f64) alignment, so
// this sidesteps that pitfall entirely rather than working around it.
//
// Mirrors `simq_state::simd::single_qubit`'s convention (the CPU reference
// this kernel is cross-validated against): amplitude index bit `q` (weight
// 2^q) holds qubit q's value. Requires wgpu::Features::SHADER_F64 (device
// creation fails loudly, per `gpu.rs`, if an adapter doesn't support it —
// no silent f32 downgrade).

struct Complex {
    re: f64,
    im: f64,
};

@group(0) @binding(0)
var<storage, read_write> state: array<Complex>;
@group(0) @binding(1)
var<storage, read> gate: array<f64, 8>; // row-major [re00,im00,re01,im01,re10,im10,re11,im11]
@group(0) @binding(2)
var<storage, read> params: array<u32, 1>; // [qubit]

fn g(r: u32, c: u32) -> Complex {
    let base = (r * 2u + c) * 2u;
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
    let half = dim >> 1u;
    let i = gid.x;
    // Explicit bound, not just the destination-index check below: dispatch
    // rounds up to a whole number of 64-wide workgroups, so invocations
    // with i >= half can exist and must not touch the state at all (their
    // `pair` value would otherwise alias an index another invocation is
    // also writing, corrupting the result).
    if i >= half {
        return;
    }

    let qubit = params[0];
    let stride = 1u << qubit;
    let pair = (i / stride) * stride * 2u + (i % stride);
    let j = pair + stride;

    let a = state[pair];
    let b = state[j];

    let new_a = cadd(cmul(g(0u, 0u), a), cmul(g(0u, 1u), b));
    let new_b = cadd(cmul(g(1u, 0u), a), cmul(g(1u, 1u), b));

    state[pair] = new_a;
    state[j] = new_b;
}
