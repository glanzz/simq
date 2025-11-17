//! Execution Plan Generation Demo
//!
//! This example demonstrates the execution plan generation feature,
//! which analyzes circuits and creates optimized execution plans with
//! parallelization opportunities and resource estimation.

use simq_compiler::{
    execution_plan::ExecutionPlanner,
    pipeline::{create_compiler, OptimizationLevel},
};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;

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
            "H" | "X" | "Y" | "Z" | "S" | "T" => 1,
            "CNOT" | "CZ" | "SWAP" => 2,
            _ => 1,
        }
    }
}

fn main() {
    println!("=== Execution Plan Generation Demo ===\n");

    // ===================================================================
    // 1. Basic Execution Plan
    // ===================================================================
    println!("=== 1. Basic Execution Plan ===");

    let circuit = create_simple_circuit();
    println!("Circuit: {} gates on {} qubits\n", circuit.len(), circuit.num_qubits());

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);

    println!("Execution Plan:");
    println!("  Depth: {} layers", plan.depth);
    println!("  Total gates: {}", plan.gate_count);
    println!("  Parallelism factor: {:.2}x", plan.parallelism_factor);
    println!("  Estimated time: {:.2} µs", plan.total_time);
    println!("  Critical path: {}", plan.critical_path_length);

    println!("\nLayer breakdown:");
    for (i, layer) in plan.layers.iter().enumerate() {
        println!("  Layer {}: {} gates, {} qubits, {:.2} µs",
            i,
            layer.gates.len(),
            layer.qubits.len(),
            layer.estimated_time
        );
    }

    println!("\nResource Requirements:");
    println!("  Peak memory: {} bytes", plan.resources.peak_memory);
    println!("  Peak qubits: {}", plan.resources.peak_qubits);
    println!("  Two-qubit gates: {}", plan.resources.two_qubit_gates);
    println!("  Measurements: {}", plan.resources.measurement_count);

    // ===================================================================
    // 2. Parallelism Analysis
    // ===================================================================
    println!("\n=== 2. Parallelism Analysis ===");

    // Create circuits with different parallelism characteristics
    let sequential = create_sequential_circuit();
    let parallel = create_parallel_circuit();
    let mixed = create_mixed_circuit();

    println!("\nSequential circuit (single qubit chain):");
    let plan_seq = planner.generate_plan(&sequential);
    print_plan_summary(&plan_seq);

    println!("\nParallel circuit (independent gates):");
    let plan_par = planner.generate_plan(&parallel);
    print_plan_summary(&plan_par);

    println!("\nMixed circuit (partial parallelism):");
    let plan_mix = planner.generate_plan(&mixed);
    print_plan_summary(&plan_mix);

    // ===================================================================
    // 3. Optimization Impact
    // ===================================================================
    println!("\n=== 3. Optimization Impact on Execution Plan ===");

    let unopt_circuit = create_circuit_with_redundancy();
    println!("Original circuit: {} gates", unopt_circuit.len());

    let unopt_plan = planner.generate_plan(&unopt_circuit);
    println!("  Depth: {}, Time: {:.2} µs", unopt_plan.depth, unopt_plan.total_time);

    // Optimize the circuit
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut opt_circuit = unopt_circuit.clone();
    compiler.compile(&mut opt_circuit).unwrap();

    println!("\nOptimized circuit: {} gates", opt_circuit.len());
    let opt_plan = planner.generate_plan(&opt_circuit);
    println!("  Depth: {}, Time: {:.2} µs", opt_plan.depth, opt_plan.total_time);

    println!("\nImprovements:");
    println!("  Gate reduction: {:.1}%",
        100.0 * (1.0 - opt_circuit.len() as f64 / unopt_circuit.len() as f64));
    println!("  Depth reduction: {:.1}%",
        100.0 * (1.0 - opt_plan.depth as f64 / unopt_plan.depth as f64));
    println!("  Time reduction: {:.1}%",
        100.0 * (1.0 - opt_plan.total_time / unopt_plan.total_time));

    // ===================================================================
    // 4. Layer Visualization
    // ===================================================================
    println!("\n=== 4. Execution Layer Visualization ===");

    let circuit = create_visualization_circuit();
    let plan = planner.generate_plan(&circuit);

    println!("Circuit with {} qubits:\n", circuit.num_qubits());
    visualize_plan(&plan, &circuit);

    // ===================================================================
    // 5. Custom Gate Timing
    // ===================================================================
    println!("\n=== 5. Custom Gate Timing ===");

    let circuit = create_mixed_gates_circuit();

    // Default timing
    let default_planner = ExecutionPlanner::new();
    let default_plan = default_planner.generate_plan(&circuit);
    println!("With default gate times:");
    println!("  Total time: {:.2} µs", default_plan.total_time);

    // Custom timing (e.g., hardware-specific)
    let mut custom_planner = ExecutionPlanner::new();
    custom_planner.set_gate_time("H", 0.05);  // Fast single-qubit
    custom_planner.set_gate_time("CNOT", 0.5);  // Slower two-qubit

    let custom_plan = custom_planner.generate_plan(&circuit);
    println!("\nWith custom gate times:");
    println!("  Total time: {:.2} µs", custom_plan.total_time);
    println!("  Difference: {:.1}%",
        100.0 * (custom_plan.total_time / default_plan.total_time - 1.0));

    // ===================================================================
    // 6. Large Circuit Analysis
    // ===================================================================
    println!("\n=== 6. Large Circuit Analysis ===");

    for size in [10, 50, 100] {
        let circuit = create_large_circuit(size);
        let plan = planner.generate_plan(&circuit);

        println!("\nCircuit with {} gates:", circuit.len());
        println!("  Depth: {} layers", plan.depth);
        println!("  Parallelism: {:.2}x", plan.parallelism_factor);
        println!("  Avg gates/layer: {:.1}", plan.gate_count as f64 / plan.depth as f64);
        println!("  Estimated time: {:.2} µs", plan.total_time);
    }

    println!("\n=== Demo Complete ===");
}

/// Print a concise summary of an execution plan
fn print_plan_summary(plan: &simq_compiler::execution_plan::ExecutionPlan) {
    println!("  Gates: {}, Depth: {}, Parallelism: {:.2}x, Time: {:.2} µs",
        plan.gate_count,
        plan.depth,
        plan.parallelism_factor,
        plan.total_time
    );
}

/// Visualize execution plan as ASCII art
fn visualize_plan(plan: &simq_compiler::execution_plan::ExecutionPlan, circuit: &Circuit) {
    let num_qubits = circuit.num_qubits();

    println!("Layer visualization (Q = qubit, G = gate):\n");

    for (layer_idx, layer) in plan.layers.iter().enumerate() {
        println!("Layer {} ({} gates, {:.2} µs):",
            layer_idx, layer.gates.len(), layer.estimated_time);

        // Show which qubits are active in this layer
        for q in 0..num_qubits {
            let qubit_id = QubitId::new(q);
            let active = layer.qubits.contains(&qubit_id);
            if active {
                print!("  Q{}: [G] ", q);
            } else {
                print!("  Q{}: [ ] ", q);
            }
        }
        println!("\n");
    }
}

/// Create a simple circuit for demonstration
fn create_simple_circuit() -> Circuit {
    let mut circuit = Circuit::new(3);

    let h = Arc::new(MockGate { name: "H".to_string() });
    let cnot = Arc::new(MockGate { name: "CNOT".to_string() });

    // Layer 1: Hadamards on all qubits (parallel)
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();

    // Layer 2: CNOTs (some parallel)
    circuit.add_gate(cnot.clone(), &[QubitId::new(0), QubitId::new(1)]).unwrap();
    circuit.add_gate(cnot.clone(), &[QubitId::new(1), QubitId::new(2)]).unwrap();

    circuit
}

/// Create a sequential circuit (no parallelism)
fn create_sequential_circuit() -> Circuit {
    let mut circuit = Circuit::new(2);

    let x = Arc::new(MockGate { name: "X".to_string() });

    // All gates on qubit 0
    for _ in 0..10 {
        circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();
    }

    circuit
}

/// Create a parallel circuit (maximum parallelism)
fn create_parallel_circuit() -> Circuit {
    let mut circuit = Circuit::new(10);

    let h = Arc::new(MockGate { name: "H".to_string() });

    // All gates on different qubits (can all run in parallel)
    for i in 0..10 {
        circuit.add_gate(h.clone(), &[QubitId::new(i)]).unwrap();
    }

    circuit
}

/// Create a circuit with mixed parallelism
fn create_mixed_circuit() -> Circuit {
    let mut circuit = Circuit::new(4);

    let h = Arc::new(MockGate { name: "H".to_string() });
    let cnot = Arc::new(MockGate { name: "CNOT".to_string() });

    // Some parallel, some sequential
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();

    circuit.add_gate(cnot.clone(), &[QubitId::new(0), QubitId::new(1)]).unwrap();
    circuit.add_gate(cnot.clone(), &[QubitId::new(2), QubitId::new(3)]).unwrap();

    circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(3)]).unwrap();

    circuit
}

/// Create a circuit with redundancy (for optimization demo)
fn create_circuit_with_redundancy() -> Circuit {
    let mut circuit = Circuit::new(3);

    let h = Arc::new(MockGate { name: "H".to_string() });
    let x = Arc::new(MockGate { name: "X".to_string() });

    // Inverse pairs (H-H, X-X)
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();  // Cancels

    circuit.add_gate(x.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(x.clone(), &[QubitId::new(1)]).unwrap();  // Cancels

    // Some useful gates
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();
    circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();

    circuit
}

/// Create a circuit for visualization
fn create_visualization_circuit() -> Circuit {
    let mut circuit = Circuit::new(4);

    let h = Arc::new(MockGate { name: "H".to_string() });
    let x = Arc::new(MockGate { name: "X".to_string() });

    // Layer 0: Q0, Q2
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();

    // Layer 1: Q1, Q3
    circuit.add_gate(x.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(x.clone(), &[QubitId::new(3)]).unwrap();

    // Layer 2: Q0, Q1, Q2
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();

    circuit
}

/// Create a circuit with mixed gate types
fn create_mixed_gates_circuit() -> Circuit {
    let mut circuit = Circuit::new(4);

    let h = Arc::new(MockGate { name: "H".to_string() });
    let t = Arc::new(MockGate { name: "T".to_string() });
    let cnot = Arc::new(MockGate { name: "CNOT".to_string() });

    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(t.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(cnot.clone(), &[QubitId::new(0), QubitId::new(1)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();
    circuit.add_gate(cnot.clone(), &[QubitId::new(2), QubitId::new(3)]).unwrap();

    circuit
}

/// Create a large circuit for performance analysis
fn create_large_circuit(num_gates: usize) -> Circuit {
    let num_qubits = (num_gates / 10).max(5);
    let mut circuit = Circuit::new(num_qubits);

    let gates = vec![
        Arc::new(MockGate { name: "H".to_string() }),
        Arc::new(MockGate { name: "X".to_string() }),
        Arc::new(MockGate { name: "T".to_string() }),
        Arc::new(MockGate { name: "CNOT".to_string() }),
    ];

    for i in 0..num_gates {
        let gate_idx = i % gates.len();
        let gate = &gates[gate_idx];

        if gate.num_qubits() == 1 {
            let qubit = i % num_qubits;
            circuit.add_gate(gate.clone(), &[QubitId::new(qubit)]).unwrap();
        } else {
            let q1 = i % num_qubits;
            let q2 = (i + 1) % num_qubits;
            if q1 != q2 {
                circuit.add_gate(gate.clone(), &[QubitId::new(q1), QubitId::new(q2)]).unwrap();
            }
        }
    }

    circuit
}
