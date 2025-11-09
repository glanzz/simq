//! Example: Circuit serialization for caching and persistence
//!
//! This example demonstrates how to serialize and deserialize quantum circuits
//! for caching and persistence.

use simq_core::circuit::Circuit;
use simq_core::gate::Gate;
use simq_core::{QubitId, Result};
use std::sync::Arc;

// Mock gate implementations for demonstration
#[derive(Debug)]
struct HadamardGate;

impl Gate for HadamardGate {
    fn name(&self) -> &str {
        "H"
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn is_hermitian(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct CnotGate;

impl Gate for CnotGate {
    fn name(&self) -> &str {
        "CNOT"
    }

    fn num_qubits(&self) -> usize {
        2
    }
}

fn main() -> Result<()> {
    println!("=== SimQ Circuit Serialization Example ===\n");

    // Create a simple Bell state circuit
    println!("1. Creating a Bell state circuit...");
    let mut circuit = Circuit::new(2);
    let h_gate = Arc::new(HadamardGate);
    let cnot_gate = Arc::new(CnotGate);

    circuit.add_gate(h_gate, &[QubitId::new(0)])?;
    circuit.add_gate(cnot_gate, &[QubitId::new(0), QubitId::new(1)])?;

    println!("   Circuit: {} qubits, {} operations", circuit.num_qubits(), circuit.len());
    println!();

    // Serialize to JSON
    println!("2. Serializing circuit to JSON...");
    let json = circuit.to_json()?;
    println!("   JSON length: {} bytes", json.len());
    println!("   JSON preview: {}", &json[..json.len().min(100)]);
    if json.len() > 100 {
        println!("   ...");
    }
    println!();

    // Serialize to binary
    println!("3. Serializing circuit to binary...");
    let bytes = circuit.to_bytes()?;
    println!("   Binary length: {} bytes", bytes.len());
    println!("   Compression ratio: {:.2}x smaller than JSON", json.len() as f64 / bytes.len() as f64);
    println!();

    // Generate cache key
    println!("4. Generating cache key...");
    let cache_key = circuit.cache_key();
    println!("   Cache key: {}", cache_key);
    println!("   (Same circuit structure will have the same cache key)");
    println!();

    // Demonstrate cache key consistency
    println!("5. Verifying cache key consistency...");
    let mut circuit2 = Circuit::new(2);
    let h_gate2 = Arc::new(HadamardGate);
    let cnot_gate2 = Arc::new(CnotGate);
    circuit2.add_gate(h_gate2, &[QubitId::new(0)])?;
    circuit2.add_gate(cnot_gate2, &[QubitId::new(0), QubitId::new(1)])?;

    let cache_key2 = circuit2.cache_key();
    println!("   Cache key 1: {}", cache_key);
    println!("   Cache key 2: {}", cache_key2);
    println!("   Keys match: {}", cache_key == cache_key2);
    println!();

    #[cfg(feature = "cache")]
    {
        use simq_core::serialization::cache::{CircuitCache, CircuitKey, MemoryCache};

        println!("6. Demonstrating cache functionality...");
        let cache = MemoryCache::new();
        let key = CircuitKey::from_circuit(&circuit);

        // Cache miss
        println!("   Checking cache (should be miss)...");
        let cached = cache.get(&key);
        println!("   Cache hit: {}", cached.is_some());

        // Put in cache
        println!("   Storing circuit in cache...");
        cache.put(key.clone(), circuit.clone())?;

        // Cache hit
        println!("   Checking cache again (should be hit)...");
        let cached = cache.get(&key);
        println!("   Cache hit: {}", cached.is_some());

        if let Some(cached_circuit) = cached {
            println!("   Retrieved circuit: {} qubits, {} operations",
                     cached_circuit.num_qubits(), cached_circuit.len());
        }

        // Get cache statistics
        let stats = cache.stats();
        println!("   Cache stats: {} hits, {} misses, {} entries",
                 stats.hits, stats.misses, stats.size);
        println!("   Hit rate: {:.2}%", stats.hit_rate());
        println!();
    }

    #[cfg(not(feature = "cache"))]
    {
        println!("6. Cache functionality not available (enable 'cache' feature)");
        println!();
    }

    println!("✓ Serialization example completed successfully!");

    Ok(())
}

