//! Demonstration of lookup table performance for small-angle rotations
//!
//! This example shows how lookup tables can significantly improve performance
//! when computing rotation gate matrices for small angles, which is common in
//! variational quantum algorithms like VQE and QAOA.

use simq_core::gate::Gate;
use simq_gates::lookup::{LookupConfig, RotationLookupTable};
use simq_gates::matrices;
use simq_gates::optimized::{
    create_compact_table, create_global_lookup_table, create_high_precision_table,
    OptimizedRotationX, OptimizedRotationY, OptimizedRotationZ,
};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("==============================================");
    println!("Lookup Table Demo for Small-Angle Rotations");
    println!("==============================================\n");

    // ========================================================================
    // Part 1: Basic Lookup Table Usage
    // ========================================================================

    println!("--- PART 1: Basic Lookup Table Usage ---\n");

    let config = LookupConfig::new()
        .max_angle(PI / 4.0)
        .num_entries(1024)
        .interpolation_enabled(true);

    let table = RotationLookupTable::new(config);

    println!("Created lookup table with configuration:");
    println!("{}\n", table.stats());

    // Test with a small angle
    let small_angle: f64 = 0.01; // ~0.57 degrees
    println!(
        "Testing with small angle: {} rad ({:.2}°)",
        small_angle,
        small_angle.to_degrees()
    );

    let matrix_lookup = table.rx_matrix(small_angle);
    let matrix_direct = matrices::rotation_x(small_angle);

    println!("\nRX matrix from lookup:");
    print_matrix(&matrix_lookup);

    println!("\nRX matrix from direct computation:");
    print_matrix(&matrix_direct);

    println!("\nMaximum element-wise error:");
    let max_error = compute_max_error(&matrix_lookup, &matrix_direct);
    println!("  {:.2e}", max_error);

    // Test with a large angle (should fall back to direct computation)
    let large_angle: f64 = PI;
    println!(
        "\n\nTesting with large angle: {} rad ({:.2}°)",
        large_angle,
        large_angle.to_degrees()
    );
    println!("(This will automatically use direct computation)");

    let matrix_large = table.rx_matrix(large_angle);
    let matrix_ref = matrices::rotation_x(large_angle);
    let error_large = compute_max_error(&matrix_large, &matrix_ref);
    println!("Max error: {:.2e} (identical - using same function)", error_large);

    // ========================================================================
    // Part 2: Pre-configured Tables
    // ========================================================================

    println!("\n\n--- PART 2: Pre-configured Tables ---\n");

    let global_table = create_global_lookup_table();
    let high_precision = create_high_precision_table();
    let compact = create_compact_table();

    println!("Global table (general use):");
    println!("{}", global_table.stats());

    println!("\nHigh precision table:");
    println!("{}", high_precision.stats());

    println!("\nCompact table (memory-constrained):");
    println!("{}", compact.stats());

    // ========================================================================
    // Part 3: Optimized Gate Structures
    // ========================================================================

    println!("\n\n--- PART 3: Optimized Gate Structures ---\n");

    let angle = 0.05;
    let rx = OptimizedRotationX::new(angle, &global_table);
    let ry = OptimizedRotationY::new(angle, &global_table);
    let rz = OptimizedRotationZ::new(angle, &global_table);

    println!("Created optimized gates with angle {} rad:", angle);
    println!("  RX: {}", rx.description());
    println!("  RY: {}", ry.description());
    println!("  RZ: {}", rz.description());

    println!("\nComputing matrices...");
    let rx_matrix = rx.compute_matrix();
    let ry_matrix = ry.compute_matrix();
    let rz_matrix = rz.compute_matrix();

    println!("\nRX({}) matrix:", angle);
    print_matrix(&rx_matrix);

    println!("\nRY({}) matrix:", angle);
    print_matrix(&ry_matrix);

    println!("\nRZ({}) matrix:", angle);
    print_matrix(&rz_matrix);

    // ========================================================================
    // Part 4: Performance Comparison
    // ========================================================================

    println!("\n\n--- PART 4: Performance Comparison ---\n");

    let num_iterations = 100_000;
    let test_angles: Vec<f64> = (0..100).map(|i| i as f64 * 0.005).collect();

    println!(
        "Running {} iterations with {} different angles",
        num_iterations,
        test_angles.len()
    );
    println!("(Total matrix computations: {})\n", num_iterations * test_angles.len());

    // Benchmark direct computation
    println!("Benchmarking direct computation (baseline)...");
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &angle in &test_angles {
            let _ = matrices::rotation_x(angle);
            let _ = matrices::rotation_y(angle);
            let _ = matrices::rotation_z(angle);
        }
    }
    let direct_time = start.elapsed();
    println!("  Time: {:?}", direct_time);

    // Benchmark lookup table (no interpolation)
    println!("\nBenchmarking lookup table (no interpolation)...");
    let config_no_interp = LookupConfig::new()
        .max_angle(PI / 2.0)
        .num_entries(2048)
        .interpolation_enabled(false);
    let table_no_interp = RotationLookupTable::new(config_no_interp);

    let start = Instant::now();
    for _ in 0..num_iterations {
        for &angle in &test_angles {
            let _ = table_no_interp.rx_matrix(angle);
            let _ = table_no_interp.ry_matrix(angle);
            let _ = table_no_interp.rz_matrix(angle);
        }
    }
    let lookup_time = start.elapsed();
    println!("  Time: {:?}", lookup_time);

    // Benchmark lookup table (with interpolation)
    println!("\nBenchmarking lookup table (with interpolation)...");
    let start = Instant::now();
    for _ in 0..num_iterations {
        for &angle in &test_angles {
            let _ = global_table.rx_matrix(angle);
            let _ = global_table.ry_matrix(angle);
            let _ = global_table.rz_matrix(angle);
        }
    }
    let interp_time = start.elapsed();
    println!("  Time: {:?}", interp_time);

    // ========================================================================
    // Part 5: Results Summary
    // ========================================================================

    println!("\n\n--- PERFORMANCE SUMMARY ---\n");

    let speedup_no_interp = direct_time.as_secs_f64() / lookup_time.as_secs_f64();
    let speedup_interp = direct_time.as_secs_f64() / interp_time.as_secs_f64();

    println!("Direct computation:           {:>10.2} ms  (baseline)", direct_time.as_millis());
    println!(
        "Lookup (no interpolation):    {:>10.2} ms  ({:.2}× speedup)",
        lookup_time.as_millis(),
        speedup_no_interp
    );
    println!(
        "Lookup (with interpolation):  {:>10.2} ms  ({:.2}× speedup)",
        interp_time.as_millis(),
        speedup_interp
    );

    println!("\n\n--- ACCURACY ANALYSIS ---\n");

    let mut max_errors_no_interp = Vec::new();
    let mut max_errors_interp = Vec::new();

    for &angle in test_angles.iter().take(20) {
        let ref_matrix = matrices::rotation_x(angle);

        let lookup_no_interp = table_no_interp.rx_matrix(angle);
        let error_no_interp = compute_max_error(&lookup_no_interp, &ref_matrix);
        max_errors_no_interp.push(error_no_interp);

        let lookup_interp = global_table.rx_matrix(angle);
        let error_interp = compute_max_error(&lookup_interp, &ref_matrix);
        max_errors_interp.push(error_interp);
    }

    let avg_error_no_interp =
        max_errors_no_interp.iter().sum::<f64>() / max_errors_no_interp.len() as f64;
    let avg_error_interp = max_errors_interp.iter().sum::<f64>() / max_errors_interp.len() as f64;

    println!("Average maximum error (no interpolation): {:.2e}", avg_error_no_interp);
    println!("Average maximum error (with interpolation): {:.2e}", avg_error_interp);
    println!(
        "\nInterpolation improves accuracy by {:.2}×",
        avg_error_no_interp / avg_error_interp
    );

    // ========================================================================
    // Part 6: Use Case Recommendation
    // ========================================================================

    println!("\n\n--- RECOMMENDATIONS ---\n");

    println!("✓ Use lookup tables when:");
    println!("  - Running VQE, QAOA, or other variational algorithms");
    println!("  - Performing gradient-based optimization");
    println!("  - Circuit has many small-angle rotations (< π/4)");
    println!("  - Performance is critical");

    println!("\n✓ Use direct computation when:");
    println!("  - Angles are uniformly distributed or mostly large");
    println!("  - Memory is extremely constrained");
    println!("  - Circuit is executed only once or very few times");

    println!("\n✓ Recommended configurations:");
    println!("  - General use: 2048 entries, max angle π/2, interpolation enabled (~32 KB)");
    println!("  - High precision: 4096 entries, max angle π/2, interpolation enabled (~64 KB)");
    println!("  - Memory-constrained: 512 entries, max angle π/4, interpolation enabled (~8 KB)");

    println!("\n==============================================");
    println!("Demo complete!");
    println!("==============================================\n");
}

fn print_matrix(matrix: &[[num_complex::Complex64; 2]; 2]) {
    for row in matrix {
        print!("  [");
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                print!(",  ");
            }
            print!("{:>8.5}{:>+8.5}i", val.re, val.im);
        }
        println!("]");
    }
}

fn compute_max_error(
    m1: &[[num_complex::Complex64; 2]; 2],
    m2: &[[num_complex::Complex64; 2]; 2],
) -> f64 {
    let mut max_error: f64 = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let error_re = (m1[i][j].re - m2[i][j].re).abs();
            let error_im = (m1[i][j].im - m2[i][j].im).abs();
            max_error = max_error.max(error_re).max(error_im);
        }
    }
    max_error
}
