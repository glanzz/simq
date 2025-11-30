//! Compilation Caching Demo
//!
//! This example demonstrates the compilation caching feature, which
//! dramatically improves performance when compiling identical or similar circuits.

use simq_compiler::{
    cache::CircuitFingerprint,
    pipeline::{create_compiler, OptimizationLevel},
    CachedCompiler, SharedCachedCompiler,
};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug)]
struct MockGate {
    name: String,
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }
    fn num_qubits(&self) -> usize {
        match self.name.as_str() {
            "H" | "X" | "Y" | "Z" => 1,
            "CNOT" | "CZ" => 2,
            _ => 1,
        }
    }
}

fn main() {
    println!("=== Compilation Caching Demo ===\n");

    // ===================================================================
    // 1. Basic Caching Example
    // ===================================================================
    println!("=== 1. Basic Caching ===");

    let compiler = create_compiler(OptimizationLevel::O2);
    let mut cached_compiler = CachedCompiler::new(compiler, 100);

    // Create first circuit
    let mut circuit1 = create_sample_circuit();
    println!("Circuit 1 fingerprint: 0x{:x}", CircuitFingerprint::compute(&circuit1).value());

    // First compilation (cache miss)
    let start = Instant::now();
    let result1 = cached_compiler.compile(&mut circuit1).unwrap();
    let time1 = start.elapsed();

    println!("First compilation:");
    println!("  Cached: {}", result1.is_cached());
    println!("  Time: {:?}", time1);
    println!("  Final gates: {}", circuit1.len());

    // Create identical circuit
    let mut circuit2 = create_sample_circuit();

    // Second compilation (cache hit!)
    let start = Instant::now();
    let result2 = cached_compiler.compile(&mut circuit2).unwrap();
    let time2 = start.elapsed();

    println!("\nSecond compilation (identical circuit):");
    println!("  Cached: {}", result2.is_cached());
    println!("  Time: {:?}", time2);
    println!("  Speedup: {:.1}x", time1.as_nanos() as f64 / time2.as_nanos() as f64);

    // Cache statistics
    println!("\n{}", cached_compiler.cache().statistics());

    // ===================================================================
    // 2. Cache Performance Comparison
    // ===================================================================
    println!("=== 2. Performance Comparison ===");

    let compiler = create_compiler(OptimizationLevel::O2);
    let mut cached = CachedCompiler::new(compiler.clone(), 50);
    let uncached = compiler;

    let num_circuits = 100;
    let num_unique = 10; // Only 10 unique circuit patterns

    println!("Compiling {} circuits ({} unique patterns)...", num_circuits, num_unique);

    // Benchmark uncached
    let start = Instant::now();
    for i in 0..num_circuits {
        let mut circuit = create_pattern_circuit(i % num_unique);
        uncached.compile(&mut circuit).unwrap();
    }
    let uncached_time = start.elapsed();

    // Benchmark cached
    let start = Instant::now();
    for i in 0..num_circuits {
        let mut circuit = create_pattern_circuit(i % num_unique);
        cached.compile(&mut circuit).unwrap();
    }
    let cached_time = start.elapsed();

    println!("\nResults:");
    println!(
        "  Uncached: {:?} ({:.2} ms/circuit)",
        uncached_time,
        uncached_time.as_secs_f64() * 1000.0 / num_circuits as f64
    );
    println!(
        "  Cached:   {:?} ({:.2} ms/circuit)",
        cached_time,
        cached_time.as_secs_f64() * 1000.0 / num_circuits as f64
    );
    println!("  Speedup:  {:.1}x", uncached_time.as_secs_f64() / cached_time.as_secs_f64());

    let stats = cached.cache().statistics();
    println!("\nCache Statistics:");
    println!("  Hit rate: {:.1}%", stats.hit_rate());
    println!("  Hits: {}", stats.hits);
    println!("  Misses: {}", stats.misses);
    println!("  Cache size: {}/{}", stats.current_size, stats.max_size);

    // ===================================================================
    // 3. LRU Eviction Demonstration
    // ===================================================================
    println!("\n=== 3. LRU Cache Eviction ===");

    let compiler = create_compiler(OptimizationLevel::O1);
    let mut cached = CachedCompiler::new(compiler, 5); // Small cache

    println!("Cache size: 5");
    println!("\nInserting 10 unique circuits...");

    for i in 0..10 {
        let mut circuit = create_pattern_circuit(i);
        cached.compile(&mut circuit).unwrap();
        println!("  Circuit {}: cache size = {}", i, cached.cache().len());
    }

    println!("\nFinal cache statistics:");
    let stats = cached.cache().statistics();
    println!("  Entries: {}", stats.current_size);
    println!("  Evictions: {}", stats.evictions);
    println!("  (Least recently used circuits were evicted)");

    // ===================================================================
    // 4. Thread-Safe Caching
    // ===================================================================
    println!("\n=== 4. Thread-Safe Shared Cache ===");

    let compiler = create_compiler(OptimizationLevel::O1);
    let shared_cache = SharedCachedCompiler::new(compiler, 20);

    println!("Compiling circuits with shared cache...");

    // Simulate multiple "threads" using the same cache
    for thread_id in 0..3 {
        let cache_clone = shared_cache.clone();

        for i in 0..10 {
            let mut circuit = create_pattern_circuit(i % 5);
            let result = cache_clone.compile(&mut circuit).unwrap();
            if result.is_cached() {
                println!("  Thread {} circuit {}: CACHE HIT", thread_id, i);
            }
        }
    }

    let stats = shared_cache.cache_statistics();
    println!("\nShared cache statistics:");
    println!("  Total compilations: {}", stats.hits + stats.misses);
    println!("  Cache hits: {}", stats.hits);
    println!("  Hit rate: {:.1}%", stats.hit_rate());

    // ===================================================================
    // 5. Cache Control
    // ===================================================================
    println!("\n=== 5. Cache Control ===");

    let compiler = create_compiler(OptimizationLevel::O2);
    let mut cached = CachedCompiler::new(compiler, 10);

    // Compile some circuits
    for i in 0..5 {
        let mut circuit = create_pattern_circuit(i);
        cached.compile(&mut circuit).unwrap();
    }
    println!("Cache size after 5 compilations: {}", cached.cache().len());

    // Disable caching
    cached.set_enabled(false);
    println!("\nCaching disabled");

    let mut circuit = create_pattern_circuit(0); // Same as first
    let result = cached.compile(&mut circuit).unwrap();
    println!("  Compiling identical circuit: cached = {}", result.is_cached());

    // Re-enable and clear
    cached.set_enabled(true);
    cached.clear_cache();
    println!("\nCache cleared");
    println!("  Cache size: {}", cached.cache().len());

    println!("\n=== Demo Complete ===");
}

/// Create a sample circuit with multiple gate patterns
fn create_sample_circuit() -> Circuit {
    let mut circuit = Circuit::new(3);

    // Inverse pairs (will be optimized away)
    let x = Arc::new(MockGate {
        name: "X".to_string(),
    });
    circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();

    // Some other gates
    let h = Arc::new(MockGate {
        name: "H".to_string(),
    });
    let z = Arc::new(MockGate {
        name: "Z".to_string(),
    });

    circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(z.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();

    circuit
}

/// Create a circuit based on a pattern number
fn create_pattern_circuit(pattern: usize) -> Circuit {
    let mut circuit = Circuit::new(2);

    let gates = vec![
        Arc::new(MockGate {
            name: "H".to_string(),
        }),
        Arc::new(MockGate {
            name: "X".to_string(),
        }),
        Arc::new(MockGate {
            name: "Y".to_string(),
        }),
        Arc::new(MockGate {
            name: "Z".to_string(),
        }),
    ];

    // Create different patterns
    for i in 0..(pattern % 5 + 1) {
        let gate = &gates[i % gates.len()];
        circuit
            .add_gate(gate.clone(), &[QubitId::new(i % 2)])
            .unwrap();
    }

    circuit
}
