// WGSL compute shader for single-qubit gate application
// Applies a 2x2 gate matrix to a dense state vector

// Buffer 0: state vector (Complex<f64> as 2 x f64)
// Buffer 1: gate matrix (4 x f64)
// Buffer 2: qubit index (u32)

struct Complex {
    re: f64,
    im: f64,
};

@group(0) @binding(0)
var<storage, read_write> state: array<Complex>;
@group(0) @binding(1)
var<uniform> gate: array<f64, 4>;
@group(0) @binding(2)
var<uniform> qubit: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let stride = 1u << qubit;
    let pair = (i / stride) * stride * 2u + (i % stride);
    let j = pair + stride;
    if j < arrayLength(&state) {
        let a = state[pair];
        let b = state[j];
        let g00 = gate[0];
        let g01 = gate[1];
        let g10 = gate[2];
        let g11 = gate[3];
        // Complex multiply and add
        let new_a = Complex(
            a.re * g00 - a.im * g01 + b.re * g10 - b.im * g11,
            a.re * g01 + a.im * g00 + b.re * g11 + b.im * g10
        );
        let new_b = Complex(
            a.re * g10 - a.im * g11 + b.re * g00 - b.im * g01,
            a.re * g11 + a.im * g10 + b.re * g01 + b.im * g00
        );
        state[pair] = new_a;
        state[j] = new_b;
    }
}
