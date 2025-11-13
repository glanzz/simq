//! Demonstrates Copy-on-Write state branching for efficient cloning
//!
//! Run with: cargo run --example cow_branching

use num_complex::Complex64;
use simq_state::CowState;

fn main() {
    println!("=== Copy-on-Write State Branching Demo ===\n");

    // Example 1: Zero-cost cloning
    println!("1. Zero-Cost Cloning:");
    let state1 = CowState::new(15).unwrap();
    println!("   Created 15-qubit state (32K amplitudes)");
    println!("   Memory: {} bytes", state1.memory_stats().shared_memory);
    println!("   Ref count: {}", state1.ref_count());

    // Clone is O(1) - just increments reference count
    let state2 = state1.clone();
    let state3 = state1.clone();

    println!("   After 2 clones:");
    println!("   Ref count: {}", state1.ref_count());
    println!("   Total memory: {} bytes (shared!)", state1.memory_stats().shared_memory);
    println!("   Overhead: {} bytes", state1.memory_stats().total_overhead);
    println!();

    // Example 2: Copy-on-Write in action
    println!("2. Copy-on-Write Behavior:");

    let base_state = CowState::new(10).unwrap();
    let mut branch1 = base_state.clone();
    let mut branch2 = base_state.clone();

    println!("   Base state + 2 branches");
    println!("   All ref counts: {}", base_state.ref_count());

    // Hadamard gate
    let h = 1.0 / 2.0_f64.sqrt();
    let hadamard = [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ];

    // First mutation triggers copy
    let stats1 = branch1.apply_single_qubit_gate(&hadamard, 0).unwrap();
    println!("   Branch1 mutated: copied={} | refs={}", stats1.copied, branch1.ref_count());
    println!("   Base state refs: {}", base_state.ref_count());

    // Second mutation also triggers copy
    let stats2 = branch2.apply_single_qubit_gate(&hadamard, 0).unwrap();
    println!("   Branch2 mutated: copied={} | refs={}", stats2.copied, branch2.ref_count());

    // Now all three are independent
    println!("   Final ref counts: base={}, b1={}, b2={}",
             base_state.ref_count(), branch1.ref_count(), branch2.ref_count());
    println!();

    // Example 3: Exploring measurement outcomes
    println!("3. Exploring Measurement Outcomes:");

    // Create superposition
    let mut superposition = CowState::new(3).unwrap();
    superposition.apply_single_qubit_gate(&hadamard, 0).unwrap();
    superposition.apply_single_qubit_gate(&hadamard, 1).unwrap();

    println!("   Created superposition state");
    println!("   Density: {:.2}%",
             superposition.get_all_probabilities().iter()
                 .filter(|&&p| p > 1e-10)
                 .count() as f64 / superposition.dimension() as f64 * 100.0);

    // Branch to explore different measurement outcomes
    let mut outcome_0 = superposition.branch();
    let mut outcome_1 = superposition.branch();

    println!("   Branched into 2 possible outcomes");
    println!("   Ref count: {}", superposition.ref_count());

    // Measure and see different paths
    let (result_0, _) = outcome_0.measure_qubit(0, 0.1).unwrap();
    let (result_1, _) = outcome_1.measure_qubit(0, 0.9).unwrap();

    println!("   Outcome branch 0: measured {}", result_0);
    println!("   Outcome branch 1: measured {}", result_1);
    println!("   Original still in superposition!");
    println!();

    // Example 4: Variational algorithm simulation
    println!("4. Variational Algorithm Pattern:");

    let initial_state = CowState::new(8).unwrap();

    // Simulate multiple parameter trials
    let angles = vec![0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0];
    let mut trial_states = Vec::new();

    for angle in &angles {
        let mut trial = initial_state.branch();

        // Apply parameterized rotation
        let cos_val = (angle / 2.0).cos();
        let sin_val = (angle / 2.0).sin();
        let rotation = [
            [Complex64::new(cos_val, 0.0), Complex64::new(0.0, -sin_val)],
            [Complex64::new(0.0, -sin_val), Complex64::new(cos_val, 0.0)],
        ];

        trial.apply_single_qubit_gate(&rotation, 0).unwrap();
        trial_states.push(trial);
    }

    println!("   Created {} trial states from 1 initial state", trial_states.len());
    println!("   Initial state refs: {} (shared until mutation)", initial_state.ref_count());
    println!("   Each trial is now independent");
    println!();

    // Example 5: Memory efficiency comparison
    println!("5. Memory Efficiency:");

    let num_qubits = 18;
    let state_memory = (1 << num_qubits) * std::mem::size_of::<Complex64>();

    println!("   For {}-qubit system:", num_qubits);
    println!("   Single state: {} MB", state_memory / 1_000_000);

    // Without CoW: 10 copies = 10x memory
    let without_cow_memory = state_memory * 10;
    println!("   10 independent copies: {} MB", without_cow_memory / 1_000_000);

    // With CoW: 10 clones = 1x memory (until mutation)
    let base = CowState::new(num_qubits).unwrap();
    let mut clones = vec![base.clone(); 10];
    let with_cow_memory = state_memory + (std::mem::size_of::<CowState>() * 10);

    println!("   10 CoW clones (shared): {} MB", with_cow_memory / 1_000_000);
    println!("   Memory savings: {}x", without_cow_memory / with_cow_memory);

    // Mutate one - only it copies
    clones[0].apply_single_qubit_gate(&hadamard, 0).unwrap();
    let after_mutation = state_memory * 2 + (std::mem::size_of::<CowState>() * 10);
    println!("   After 1 mutation: {} MB (only 1 copy made)", after_mutation / 1_000_000);
    println!();

    // Example 6: Checkpointing workflow
    println!("6. Checkpointing Workflow:");

    let mut circuit_state = CowState::new(6).unwrap();

    // Checkpoint before applying gates
    let checkpoint = circuit_state.clone();
    println!("   Created checkpoint (ref_count={})", checkpoint.ref_count());

    // Apply some gates
    for qubit in 0..3 {
        circuit_state.apply_single_qubit_gate(&hadamard, qubit).unwrap();
    }

    println!("   Applied 3 gates to circuit");
    println!("   Circuit refs: {}, Checkpoint refs: {}",
             circuit_state.ref_count(), checkpoint.ref_count());

    // Can restore from checkpoint
    let mut restored = checkpoint.clone();
    println!("   Restored from checkpoint");
    println!("   Checkpoint still has: {} refs", checkpoint.ref_count());
    println!();

    // Example 7: Performance comparison
    println!("7. Performance Comparison:");

    let num_clones = 100;
    let qubits = 12;

    use std::time::Instant;

    // Benchmark CoW cloning
    let start = Instant::now();
    let state = CowState::new(qubits).unwrap();
    let mut cow_clones = Vec::with_capacity(num_clones);
    for _ in 0..num_clones {
        cow_clones.push(state.clone());
    }
    let cow_time = start.elapsed();

    println!("   {} CoW clones of {}-qubit state:", num_clones, qubits);
    println!("   Time: {:.2}ms (O(1) per clone)", cow_time.as_secs_f64() * 1000.0);
    println!("   Memory: {}x states, 1x actual data", num_clones + 1);
    println!();

    // Example 8: Best practices
    println!("8. Best Practices:");
    println!("   ✓ Use CowState for variational algorithms (VQE, QAOA)");
    println!("   ✓ Branch to explore measurement outcomes without recomputation");
    println!("   ✓ Checkpoint before expensive operations");
    println!("   ✓ Share read-only states across threads (Arc is thread-safe)");
    println!("   ✓ Mutation automatically copies - no manual management needed");
    println!("   ✓ Memory efficient for sparse mutation patterns");
    println!();

    println!("=== Demo Complete ===");
}
