//! Benchmarks comparing compile-time caching strategies
//!
//! This benchmark suite measures the performance of different gate matrix
//! caching approaches:
//! - Direct computation (baseline)
//! - Common angle cache (compile-time constants)
//! - VQE range cache (compile-time array)
//! - Generated angle cache (build-time generation)
//! - Enhanced universal cache (multi-level)
//! - Runtime caching (lazy evaluation)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_gates::{
    compile_time_cache::{CommonAngles, UniversalCache, VQEAngles},
    generated::{EnhancedUniversalCache, GeneratedAngleCache},
    matrices,
};
use std::f64::consts::PI;

fn bench_direct_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_computation");

    let angles = vec![PI / 4.0, PI / 2.0, PI / 8.0, 0.1, 0.01];

    for angle in angles {
        group.bench_with_input(
            BenchmarkId::new("RX", format!("{:.6}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(matrices::rotation_x(black_box(angle))));
            },
        );
    }

    group.finish();
}

fn bench_common_angles_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_angles_cache");

    // Test exact matches (should be zero-cost)
    let common_angles = vec![("PI_OVER_4", PI / 4.0), ("PI_OVER_2", PI / 2.0), ("PI", PI)];

    for (name, angle) in common_angles {
        group.bench_with_input(BenchmarkId::new("RX", name), &angle, |b, &angle| {
            b.iter(|| {
                if let Some(matrix) = CommonAngles::rx_lookup(black_box(angle)) {
                    black_box(matrix)
                } else {
                    black_box(matrices::rotation_x(angle))
                }
            });
        });
    }

    group.finish();
}

fn bench_vqe_range_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_range_cache");

    // Test angles within VQE range
    let vqe_angles = vec![0.01, 0.05, 0.1, 0.2, 0.5];

    for angle in vqe_angles {
        group.bench_with_input(
            BenchmarkId::new("RX", format!("{:.3}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(VQEAngles::rx_cached(black_box(angle))));
            },
        );
    }

    group.finish();
}

fn bench_generated_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("generated_cache");

    // Test Clifford+T angles
    let clifford_angles = vec![
        ("PI_OVER_8", PI / 8.0),
        ("PI_OVER_16", PI / 16.0),
        ("PI_OVER_32", PI / 32.0),
    ];

    for (name, angle) in clifford_angles {
        group.bench_with_input(BenchmarkId::new("RX_Clifford", name), &angle, |b, &angle| {
            b.iter(|| {
                if let Some(matrix) = GeneratedAngleCache::rx_clifford_t(black_box(angle)) {
                    black_box(matrix)
                } else {
                    black_box(matrices::rotation_x(angle))
                }
            });
        });
    }

    // Test QAOA cache
    let qaoa_angles = vec![0.5, 1.0, 1.5, 2.0];

    for angle in qaoa_angles {
        group.bench_with_input(
            BenchmarkId::new("RX_QAOA", format!("{:.2}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(GeneratedAngleCache::rx_qaoa(black_box(angle))));
            },
        );
    }

    group.finish();
}

fn bench_universal_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("universal_cache");

    let test_angles = vec![
        ("Common_PI_4", PI / 4.0), // Should hit Level 1
        ("VQE_0.1", 0.1),          // Should hit Level 2
        ("QAOA_1.0", 1.0),         // Should hit Level 3
        ("Large_5.0", 5.0),        // Should fallback
    ];

    for (name, angle) in test_angles {
        group.bench_with_input(BenchmarkId::new("RX", name), &angle, |b, &angle| {
            b.iter(|| black_box(UniversalCache::rx(black_box(angle))));
        });
    }

    group.finish();
}

fn bench_enhanced_universal_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("enhanced_universal_cache");

    let test_angles = vec![
        ("Common_PI_4", PI / 4.0),   // Level 1: Common angles
        ("Clifford_PI_8", PI / 8.0), // Level 2: Clifford+T
        ("PiFrac_PI_3", PI / 3.0),   // Level 3: π fractions
        ("VQE_0.05", 0.05),          // Level 4: VQE range
        ("QAOA_2.0", 2.0),           // Level 5: QAOA range
        ("Fallback_10.0", 10.0),     // Level 6: Runtime compute
    ];

    for (name, angle) in test_angles {
        group.bench_with_input(BenchmarkId::new("RX", name), &angle, |b, &angle| {
            b.iter(|| black_box(EnhancedUniversalCache::rx(black_box(angle))));
        });
    }

    group.finish();
}

fn bench_cache_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_comparison");

    // Compare all strategies for a typical VQE angle
    let angle = 0.1;

    group.bench_function("Direct_Computation", |b| {
        b.iter(|| black_box(matrices::rotation_x(black_box(angle))));
    });

    group.bench_function("VQE_Cache", |b| {
        b.iter(|| black_box(VQEAngles::rx_cached(black_box(angle))));
    });

    group.bench_function("Universal_Cache", |b| {
        b.iter(|| black_box(UniversalCache::rx(black_box(angle))));
    });

    group.bench_function("Enhanced_Universal_Cache", |b| {
        b.iter(|| black_box(EnhancedUniversalCache::rx(black_box(angle))));
    });

    group.finish();
}

fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access");

    // Simulate realistic workload: random angle access
    let angles: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.001) % (PI / 4.0)).collect();

    group.bench_function("Sequential_Direct", |b| {
        let mut idx = 0;
        b.iter(|| {
            let angle = angles[idx % angles.len()];
            idx += 1;
            black_box(matrices::rotation_x(black_box(angle)))
        });
    });

    group.bench_function("Sequential_Enhanced_Cache", |b| {
        let mut idx = 0;
        b.iter(|| {
            let angle = angles[idx % angles.len()];
            idx += 1;
            black_box(EnhancedUniversalCache::rx(black_box(angle)))
        });
    });

    group.finish();
}

fn bench_common_gates_vs_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("common_gates_comparison");

    // Compare accessing pre-computed matrices vs cached lookups
    group.bench_function("Hadamard_Direct_Access", |b| {
        b.iter(|| black_box(&matrices::HADAMARD));
    });

    group.bench_function("RX_PI_2_Common_Cache", |b| {
        b.iter(|| black_box(CommonAngles::rx_pi_over_2()));
    });

    group.bench_function("RX_PI_2_Lookup", |b| {
        b.iter(|| black_box(CommonAngles::rx_lookup(PI / 2.0)));
    });

    group.bench_function("RX_PI_2_Enhanced_Cache", |b| {
        b.iter(|| black_box(EnhancedUniversalCache::rx(PI / 2.0)));
    });

    group.bench_function("RX_PI_2_Direct_Compute", |b| {
        b.iter(|| black_box(matrices::rotation_x(PI / 2.0)));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_computation,
    bench_common_angles_cache,
    bench_vqe_range_cache,
    bench_generated_cache,
    bench_universal_cache,
    bench_enhanced_universal_cache,
    bench_cache_comparison,
    bench_memory_access_patterns,
    bench_common_gates_vs_cache,
);

criterion_main!(benches);
