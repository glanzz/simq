//! Advanced Compilation Pipeline Demo
//!
//! This example demonstrates the advanced compilation features including:
//! - Adaptive pass selection
//! - Multi-level optimization
//! - Hardware-aware compilation
//! - Circuit pattern analysis

use simq_compiler::{
    adaptive_pipeline::{AdaptiveCompiler, MultiLevelOptimizer},
    circuit_analysis_pass::CircuitCharacteristics,
    hardware_aware::{
        CostModel, GoogleHardware, HardwareModel, HardwareType, IBMHardware, IonQHardware,
    },
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
            "H" | "X" | "Y" | "Z" | "S" | "T" | "RX" | "RY" | "RZ" => 1,
            "CNOT" | "CZ" | "SWAP" => 2,
            _ => 1,
        }
    }
}

fn main() {
    println!("=== Advanced Compilation Pipeline Demo ===\n");

    // Create a sample circuit
    let circuit = create_sample_circuit();

    println!("Initial circuit:");
    println!("  Gates: {}", circuit.len());
    println!("  Qubits: {}\n", circuit.num_qubits());

    // ===================================================================
    // 1. Circuit Pattern Analysis
    // ===================================================================
    println!("=== 1. Circuit Pattern Analysis ===");
    let chars = CircuitCharacteristics::analyze(&circuit);

    println!("Circuit Characteristics:");
    println!("  Gate count: {}", chars.gate_count);
    println!("  Depth: {}", chars.depth);
    println!("  Single/two-qubit ratio: {:.2}", chars.single_to_two_qubit_ratio);
    println!("  Commutation density: {:.2}%", chars.commutation_density * 100.0);
    println!("  Fusion density: {:.2}%", chars.fusion_density * 100.0);
    println!("  Template density: {:.2}%", chars.template_density * 100.0);
    println!("  Dead code density: {:.2}%", chars.dead_code_density * 100.0);
    println!("  Size category: {:?}", chars.size_category());
    println!("  Suggested iterations: {}", chars.suggest_iterations());

    println!("\nPass Recommendations:");
    println!("  Use DCE: {}", chars.should_use_dce());
    println!("  Use Commutation: {}", chars.should_use_commutation());
    println!("  Use Templates: {}", chars.should_use_templates());
    println!("  Use Fusion: {}\n", chars.should_use_fusion());

    // ===================================================================
    // 2. Adaptive Compilation
    // ===================================================================
    println!("=== 2. Adaptive Compilation ===");
    let mut adaptive_circuit = create_sample_circuit();

    let adaptive = AdaptiveCompiler::new().with_verbose(true);
    let compiler = adaptive.create_for_circuit(&adaptive_circuit);
    println!("\nCompiling with adaptive compiler...");
    let result = compiler.compile(&mut adaptive_circuit).unwrap();

    println!("\nResults:");
    println!("  Final gates: {}", adaptive_circuit.len());
    println!("  Modified: {}", result.modified);
    println!("  Total time: {} µs\n", result.total_time_us);

    // ===================================================================
    // 3. Multi-Level Optimization
    // ===================================================================
    println!("=== 3. Multi-Level Optimization ===");
    let mut multilevel_circuit = create_sample_circuit();

    let optimizer = MultiLevelOptimizer::new().with_verbose(true);
    optimizer.optimize(&mut multilevel_circuit);

    println!("\nFinal gates: {}\n", multilevel_circuit.len());

    // ===================================================================
    // 4. Hardware-Aware Compilation
    // ===================================================================
    println!("=== 4. Hardware-Aware Compilation ===");
    let test_circuit = create_mixed_circuit();

    // Compare costs across different hardware platforms
    let hardware_types = vec![HardwareType::IBM, HardwareType::Google, HardwareType::IonQ];

    println!("Circuit cost comparison:");
    for hw_type in hardware_types {
        let cost_model = CostModel::new(hw_type);
        let cost = cost_model.circuit_cost(&test_circuit);
        println!("  {}: {:.2} cost units", hw_type.name(), cost);
    }

    // Detailed hardware model information
    println!("\n=== Hardware Model Details ===");

    println!("\nIBM Quantum:");
    let ibm = IBMHardware::new();
    println!("  Platform: {}", ibm.name());
    println!("  CNOT cost: {}", ibm.gate_cost_by_name("CNOT"));
    println!("  CZ cost: {}", ibm.gate_cost_by_name("CZ"));
    println!("  RZ cost: {}", ibm.gate_cost_by_name("RZ"));
    println!("  CNOT native: {}", ibm.is_native("CNOT"));
    println!("  CZ native: {}", ibm.is_native("CZ"));

    println!("\nGoogle Sycamore:");
    let google = GoogleHardware::new();
    println!("  Platform: {}", google.name());
    println!("  CNOT cost: {}", google.gate_cost_by_name("CNOT"));
    println!("  CZ cost: {}", google.gate_cost_by_name("CZ"));
    println!("  iSWAP cost: {}", google.gate_cost_by_name("iSWAP"));
    println!("  CZ native: {}", google.is_native("CZ"));
    println!("  CNOT native: {}", google.is_native("CNOT"));

    println!("\nIonQ Trapped-Ion:");
    let ionq = IonQHardware::new();
    println!("  Platform: {}", ionq.name());
    println!("  XX cost: {}", ionq.gate_cost_by_name("XX"));
    println!("  CNOT cost: {}", ionq.gate_cost_by_name("CNOT"));
    println!("  RX cost: {}", ionq.gate_cost_by_name("RX"));

    // ===================================================================
    // 5. Standard Optimization Levels
    // ===================================================================
    println!("\n=== 5. Standard Optimization Level Comparison ===");

    let levels = vec![
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ];

    for level in levels {
        let mut test_circuit = create_sample_circuit();
        let initial_gates = test_circuit.len();

        let compiler = create_compiler(level);
        let result = compiler.compile(&mut test_circuit).unwrap();

        println!("{:?}:", level);
        println!("  Initial gates: {}", initial_gates);
        println!("  Final gates: {}", test_circuit.len());
        println!(
            "  Reduction: {:.1}%",
            (initial_gates - test_circuit.len()) as f64 / initial_gates as f64 * 100.0
        );
        println!("  Time: {} µs", result.total_time_us);
    }

    println!("\n=== Demo Complete ===");
}

/// Create a sample circuit with various gate patterns
fn create_sample_circuit() -> Circuit {
    let mut circuit = Circuit::new(3);

    // Add some inverse pairs (for DCE)
    let x = Arc::new(MockGate {
        name: "X".to_string(),
    });
    circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();

    // Add diagonal gates (for commutation)
    let z = Arc::new(MockGate {
        name: "Z".to_string(),
    });
    let s = Arc::new(MockGate {
        name: "S".to_string(),
    });
    let t = Arc::new(MockGate {
        name: "T".to_string(),
    });

    circuit.add_gate(z.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(s.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(t.clone(), &[QubitId::new(1)]).unwrap();

    // Add fuseable gates
    let h = Arc::new(MockGate {
        name: "H".to_string(),
    });
    circuit.add_gate(h.clone(), &[QubitId::new(2)]).unwrap();
    circuit.add_gate(s.clone(), &[QubitId::new(2)]).unwrap();

    // Add template pattern (H-Z-H = X)
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(z.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();

    circuit
}

/// Create a circuit with mixed gate types for hardware cost analysis
fn create_mixed_circuit() -> Circuit {
    let mut circuit = Circuit::new(3);

    // Single-qubit gates
    let h = Arc::new(MockGate {
        name: "H".to_string(),
    });
    let rz = Arc::new(MockGate {
        name: "RZ".to_string(),
    });
    let rx = Arc::new(MockGate {
        name: "RX".to_string(),
    });

    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(rz.clone(), &[QubitId::new(1)]).unwrap();
    circuit.add_gate(rx.clone(), &[QubitId::new(2)]).unwrap();

    // Two-qubit gates
    let cnot = Arc::new(MockGate {
        name: "CNOT".to_string(),
    });
    let cz = Arc::new(MockGate {
        name: "CZ".to_string(),
    });

    circuit
        .add_gate(cnot.clone(), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(cz.clone(), &[QubitId::new(1), QubitId::new(2)])
        .unwrap();

    // More single-qubit gates
    circuit.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
    circuit.add_gate(rz.clone(), &[QubitId::new(1)]).unwrap();

    circuit
}
