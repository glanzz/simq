//! Build script for compile-time gate matrix generation
//!
//! This script generates additional Rust code at build time for pre-computed
//! rotation gate matrices. The generated code is included in the compiled binary
//! for zero-cost access.

use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_gates.rs");
    let mut f = File::create(&dest_path).unwrap();

    // Generate header
    writeln!(
        f,
        "// Auto-generated at build time - DO NOT EDIT\n\
         // Generated rotation gate matrices for common quantum computing angles\n"
    )
    .unwrap();

    // Generate caches for different use cases
    generate_clifford_angles(&mut f);
    generate_vqe_angles(&mut f);
    generate_qaoa_angles(&mut f);
    generate_common_fractions_of_pi(&mut f);

    println!("cargo:rerun-if-changed=build.rs");
}

/// Generate matrices for Clifford+T gate angles
fn generate_clifford_angles(f: &mut File) {
    writeln!(
        f,
        "/// Pre-computed matrices for Clifford+T angles\n\
         pub mod clifford_t {{"
    )
    .unwrap();

    writeln!(f, "    use num_complex::Complex64;\n").unwrap();

    // Important angles in Clifford+T hierarchy:
    // π/2, π/4, π/8, π/16, etc.
    let angles = vec![
        ("PI_OVER_2", PI / 2.0),
        ("PI_OVER_4", PI / 4.0),
        ("PI_OVER_8", PI / 8.0),
        ("PI_OVER_16", PI / 16.0),
        ("PI_OVER_32", PI / 32.0),
    ];

    for (name, angle) in &angles {
        generate_rotation_matrix(f, "RX", name, *angle);
        generate_rotation_matrix(f, "RY", name, *angle);
        generate_rotation_matrix(f, "RZ", name, *angle);
    }

    writeln!(f, "}}").unwrap();
}

/// Generate matrices for VQE typical parameter ranges
fn generate_vqe_angles(f: &mut File) {
    writeln!(
        f,
        "\n/// Pre-computed matrices for VQE parameter ranges\n\
         pub mod vqe_params {{"
    )
    .unwrap();

    writeln!(f, "    use num_complex::Complex64;\n").unwrap();

    // Small angles commonly seen in VQE parameter updates
    let steps = 50;
    let max_angle = PI / 8.0; // Common learning rate range

    writeln!(
        f,
        "    /// Number of pre-computed VQE matrices\n\
         pub const NUM_VQE_ANGLES: usize = {};",
        steps
    )
    .unwrap();

    writeln!(f, "\n    /// Pre-computed RX matrices for VQE").unwrap();
    writeln!(f, "    pub const RX_VQE: [[[Complex64; 2]; 2]; {}] = [", steps).unwrap();

    for i in 0..steps {
        let angle = max_angle * (i as f64) / ((steps - 1) as f64);
        let matrix = compute_rx_matrix(angle);
        write_matrix_array_element(f, &matrix, 8);
        if i < steps - 1 {
            writeln!(f, ",").unwrap();
        }
    }
    writeln!(f, "\n    ];").unwrap();

    writeln!(f, "}}").unwrap();
}

/// Generate matrices for QAOA typical angles
fn generate_qaoa_angles(f: &mut File) {
    writeln!(
        f,
        "\n/// Pre-computed matrices for QAOA angles\n\
         pub mod qaoa {{"
    )
    .unwrap();

    writeln!(f, "    use num_complex::Complex64;\n").unwrap();

    // QAOA mixer and cost Hamiltonian typical angles
    // Usually in range [0, π]
    let steps = 100;
    let max_angle = PI;

    writeln!(
        f,
        "    /// Number of pre-computed QAOA matrices\n\
         pub const NUM_QAOA_ANGLES: usize = {};",
        steps
    )
    .unwrap();

    writeln!(f, "\n    /// Angle step size").unwrap();
    writeln!(f, "    pub const ANGLE_STEP: f64 = {};", max_angle / ((steps - 1) as f64)).unwrap();

    writeln!(f, "\n    /// Pre-computed RX matrices for QAOA mixer").unwrap();
    writeln!(f, "    pub const RX_MIXER: [[[Complex64; 2]; 2]; {}] = [", steps).unwrap();

    for i in 0..steps {
        let angle = max_angle * (i as f64) / ((steps - 1) as f64);
        let matrix = compute_rx_matrix(angle);
        write_matrix_array_element(f, &matrix, 8);
        if i < steps - 1 {
            writeln!(f, ",").unwrap();
        }
    }
    writeln!(f, "\n    ];").unwrap();

    writeln!(f, "\n    /// Pre-computed RZ matrices for QAOA cost").unwrap();
    writeln!(f, "    pub const RZ_COST: [[[Complex64; 2]; 2]; {}] = [", steps).unwrap();

    for i in 0..steps {
        let angle = max_angle * (i as f64) / ((steps - 1) as f64);
        let matrix = compute_rz_matrix(angle);
        write_matrix_array_element(f, &matrix, 8);
        if i < steps - 1 {
            writeln!(f, ",").unwrap();
        }
    }
    writeln!(f, "\n    ];").unwrap();

    writeln!(f, "}}").unwrap();
}

/// Generate matrices for common fractions of π
fn generate_common_fractions_of_pi(f: &mut File) {
    writeln!(
        f,
        "\n/// Pre-computed matrices for common fractions of π\n\
         pub mod pi_fractions {{"
    )
    .unwrap();

    writeln!(f, "    use num_complex::Complex64;\n").unwrap();

    // Common fractions: π/n for n = 2, 3, 4, 5, 6, 8, 10, 12
    let fractions = vec![
        ("PI_OVER_2", 2),
        ("PI_OVER_3", 3),
        ("PI_OVER_4", 4),
        ("PI_OVER_5", 5),
        ("PI_OVER_6", 6),
        ("PI_OVER_8", 8),
        ("PI_OVER_10", 10),
        ("PI_OVER_12", 12),
    ];

    for (name, divisor) in &fractions {
        let angle = PI / (*divisor as f64);
        generate_rotation_matrix(f, "RX", name, angle);
        generate_rotation_matrix(f, "RY", name, angle);
        generate_rotation_matrix(f, "RZ", name, angle);
    }

    writeln!(f, "}}").unwrap();
}

/// Generate a single rotation matrix constant
fn generate_rotation_matrix(f: &mut File, gate: &str, name: &str, angle: f64) {
    let matrix = match gate {
        "RX" => compute_rx_matrix(angle),
        "RY" => compute_ry_matrix(angle),
        "RZ" => compute_rz_matrix(angle),
        _ => panic!("Unknown gate type: {}", gate),
    };

    writeln!(
        f,
        "    /// {}({}) - Angle: {} radians ({:.2}°)",
        gate,
        name,
        angle,
        angle.to_degrees()
    )
    .unwrap();

    // Generate inline matrix literal
    let re00 = format_f64(matrix[0][0].re);
    let im00 = format_f64(matrix[0][0].im);
    let re01 = format_f64(matrix[0][1].re);
    let im01 = format_f64(matrix[0][1].im);
    let re10 = format_f64(matrix[1][0].re);
    let im10 = format_f64(matrix[1][0].im);
    let re11 = format_f64(matrix[1][1].re);
    let im11 = format_f64(matrix[1][1].im);

    writeln!(f, "    pub const {}_{}: [[Complex64; 2]; 2] = [", gate, name).unwrap();
    writeln!(f, "        [").unwrap();
    writeln!(f, "            Complex64::new({}, {}),", re00, im00).unwrap();
    writeln!(f, "            Complex64::new({}, {}),", re01, im01).unwrap();
    writeln!(f, "        ],").unwrap();
    writeln!(f, "        [").unwrap();
    writeln!(f, "            Complex64::new({}, {}),", re10, im10).unwrap();
    writeln!(f, "            Complex64::new({}, {}),", re11, im11).unwrap();
    writeln!(f, "        ],").unwrap();
    writeln!(f, "    ];\n").unwrap();
}

/// Compute RX(θ) matrix
fn compute_rx_matrix(theta: f64) -> [[Complex<f64>; 2]; 2] {
    let half = theta / 2.0;
    let c = half.cos();
    let s = half.sin();

    [
        [Complex::new(c, 0.0), Complex::new(0.0, -s)],
        [Complex::new(0.0, -s), Complex::new(c, 0.0)],
    ]
}

/// Compute RY(θ) matrix
fn compute_ry_matrix(theta: f64) -> [[Complex<f64>; 2]; 2] {
    let half = theta / 2.0;
    let c = half.cos();
    let s = half.sin();

    [
        [Complex::new(c, 0.0), Complex::new(-s, 0.0)],
        [Complex::new(s, 0.0), Complex::new(c, 0.0)],
    ]
}

/// Compute RZ(θ) matrix
fn compute_rz_matrix(theta: f64) -> [[Complex<f64>; 2]; 2] {
    let half = theta / 2.0;
    let c = half.cos();
    let s = half.sin();

    [
        [Complex::new(c, -s), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(c, s)],
    ]
}

/// Write a 2x2 complex matrix as an array element
fn write_matrix_array_element(f: &mut File, matrix: &[[Complex<f64>; 2]; 2], indent: usize) {
    let spaces = " ".repeat(indent);

    // Format all matrix elements
    let re00 = format_f64(matrix[0][0].re);
    let im00 = format_f64(matrix[0][0].im);
    let re01 = format_f64(matrix[0][1].re);
    let im01 = format_f64(matrix[0][1].im);
    let re10 = format_f64(matrix[1][0].re);
    let im10 = format_f64(matrix[1][0].im);
    let re11 = format_f64(matrix[1][1].re);
    let im11 = format_f64(matrix[1][1].im);

    // Write as a properly nested array
    writeln!(f, "{}[", spaces).unwrap();
    writeln!(f, "{}    [", spaces).unwrap();
    writeln!(f, "{}        Complex64::new({}, {}),", spaces, re00, im00).unwrap();
    writeln!(f, "{}        Complex64::new({}, {}),", spaces, re01, im01).unwrap();
    writeln!(f, "{}    ],", spaces).unwrap();
    writeln!(f, "{}    [", spaces).unwrap();
    writeln!(f, "{}        Complex64::new({}, {}),", spaces, re10, im10).unwrap();
    writeln!(f, "{}        Complex64::new({}, {}),", spaces, re11, im11).unwrap();
    writeln!(f, "{}    ],", spaces).unwrap();
    write!(f, "{}]", spaces).unwrap();
}

/// Format f64 to ensure it's a valid Rust float literal
fn format_f64(value: f64) -> String {
    if value == 0.0 {
        "0.0".to_string()
    } else if value.is_nan() {
        "f64::NAN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "f64::INFINITY".to_string()
        } else {
            "f64::NEG_INFINITY".to_string()
        }
    } else {
        // Format with enough precision and ensure it's a float literal
        let s = format!("{:.17}", value);
        // Ensure it has a decimal point
        if !s.contains('.') && !s.contains('e') {
            format!("{}.0", s)
        } else {
            s
        }
    }
}

/// Simple complex number implementation for build script
#[derive(Debug, Clone, Copy)]
struct Complex<T> {
    re: T,
    im: T,
}

impl<T> Complex<T> {
    fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}
