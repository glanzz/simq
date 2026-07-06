use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simq_gates::lookup::{LookupConfig, RotationLookupTable};
use simq_gates::matrices;
use simq_gates::optimized::create_global_lookup_table;
use std::f64::consts::PI;

fn benchmark_rotation_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_gates_direct");

    let angles = vec![0.01, 0.05, 0.1, PI / 4.0, PI / 2.0, PI];

    for angle in angles {
        group.bench_with_input(
            BenchmarkId::new("RX", format!("{:.4}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(matrices::rotation_x(angle)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RY", format!("{:.4}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(matrices::rotation_y(angle)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RZ", format!("{:.4}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(matrices::rotation_z(angle)));
            },
        );
    }

    group.finish();
}

fn benchmark_rotation_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_gates_lookup");

    let table = create_global_lookup_table();
    let angles = vec![0.01, 0.05, 0.1, PI / 4.0, PI / 2.0, PI];

    for angle in angles {
        group.bench_with_input(
            BenchmarkId::new("RX", format!("{:.4}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(table.rx_matrix(angle)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RY", format!("{:.4}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(table.ry_matrix(angle)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("RZ", format!("{:.4}", angle)),
            &angle,
            |b, &angle| {
                b.iter(|| black_box(table.rz_matrix(angle)));
            },
        );
    }

    group.finish();
}

fn benchmark_rotation_lookup_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_table_configs");

    let configs = vec![
        ("512_entries", LookupConfig::new().num_entries(512).max_angle(PI / 4.0)),
        ("1024_entries", LookupConfig::new().num_entries(1024).max_angle(PI / 2.0)),
        ("2048_entries", LookupConfig::new().num_entries(2048).max_angle(PI / 2.0)),
        ("4096_entries", LookupConfig::new().num_entries(4096).max_angle(PI / 2.0)),
    ];

    let angle = 0.05; // Small angle that benefits from lookup

    for (name, config) in configs {
        let table = RotationLookupTable::new(config);

        group.bench_with_input(BenchmarkId::new("RX", name), &angle, |b, &angle| {
            b.iter(|| black_box(table.rx_matrix(angle)));
        });
    }

    group.finish();
}

fn benchmark_interpolation_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation_impact");

    let config_no_interp = LookupConfig::new()
        .num_entries(1024)
        .interpolation_enabled(false);

    let config_with_interp = LookupConfig::new()
        .num_entries(1024)
        .interpolation_enabled(true);

    let table_no_interp = RotationLookupTable::new(config_no_interp);
    let table_with_interp = RotationLookupTable::new(config_with_interp);

    let angle = 0.05;

    group.bench_function("no_interpolation", |b| {
        b.iter(|| black_box(table_no_interp.rx_matrix(angle)));
    });

    group.bench_function("with_interpolation", |b| {
        b.iter(|| black_box(table_with_interp.rx_matrix(angle)));
    });

    group.finish();
}

fn benchmark_vqe_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("vqe_workload");

    // Simulate a typical VQE circuit: many small-angle rotations
    let num_gates = 100;
    let angles: Vec<f64> = (0..num_gates).map(|i| 0.01 + (i as f64 * 0.001)).collect();

    group.bench_function("direct_computation", |b| {
        b.iter(|| {
            for &angle in &angles {
                black_box(matrices::rotation_x(angle));
                black_box(matrices::rotation_y(angle));
            }
        });
    });

    let table = create_global_lookup_table();

    group.bench_function("lookup_table", |b| {
        b.iter(|| {
            for &angle in &angles {
                black_box(table.rx_matrix(angle));
                black_box(table.ry_matrix(angle));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_rotation_direct,
    benchmark_rotation_lookup,
    benchmark_rotation_lookup_configs,
    benchmark_interpolation_impact,
    benchmark_vqe_workload
);
criterion_main!(benches);
