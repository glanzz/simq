//! Demonstration of the configurable optimization pipeline
//!
//! This example shows different ways to create and use optimization pipelines.

use simq_compiler::{create_compiler, OptimizationLevel, PipelineBuilder};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;
use num_complex::Complex64;

// Simple gate implementation for demo
#[derive(Debug)]
struct DemoGate {
    name: String,
    matrix: Vec<Complex64>,
}

impl Gate for DemoGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        1
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix.clone())
    }
}

fn pauli_x() -> Arc<DemoGate> {
    Arc::new(DemoGate {
        name: "X".to_string(),
        matrix: vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    })
}

fn hadamard() -> Arc<DemoGate> {
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    Arc::new(DemoGate {
        name: "H".to_string(),
        matrix: vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ],
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SimQ Optimization Pipeline Demo ===\n");

    // Create a test circuit with redundant gates
    let mut circuit = Circuit::new(3);
    let x = pauli_x();
    let h = hadamard();

    // Add gates with various optimization opportunities
    circuit.add_gate(x.clone(), &[QubitId::new(0)])?;
    circuit.add_gate(x.clone(), &[QubitId::new(0)])?; // X-X cancels
    circuit.add_gate(h.clone(), &[QubitId::new(1)])?;
    circuit.add_gate(h.clone(), &[QubitId::new(1)])?; // H-H cancels
    circuit.add_gate(x.clone(), &[QubitId::new(2)])?;
    circuit.add_gate(h.clone(), &[QubitId::new(2)])?;
    circuit.add_gate(h.clone(), &[QubitId::new(2)])?; // H-H cancels
    circuit.add_gate(x.clone(), &[QubitId::new(2)])?; // X-X cancels with first X on q2

    println!("Original circuit: {} gates", circuit.len());
    println!("{}\n", circuit);

    // Example 1: Using optimization levels
    println!("--- Example 1: Optimization Levels ---");

    for level in [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ] {
        let mut test_circuit = circuit.clone();
        let compiler = create_compiler(level);
        let result = compiler.compile(&mut test_circuit)?;

        println!(
            "{:?}: {} gates → {} gates ({}μs)",
            level,
            circuit.len(),
            test_circuit.len(),
            result.total_time_us
        );
    }

    // Example 2: Using PipelineBuilder for custom configuration
    println!("\n--- Example 2: Custom Pipeline ---");

    let mut custom_circuit = circuit.clone();
    let custom_compiler = PipelineBuilder::new()
        .with_dead_code_elimination()
        .with_gate_fusion()
        .max_iterations(5)
        .enable_timing(true)
        .build();

    let result = custom_compiler.compile(&mut custom_circuit)?;
    println!("Custom pipeline: {} gates → {} gates", circuit.len(), custom_circuit.len());
    println!("Statistics:");
    for stat in &result.pass_stats {
        println!("  - {}: {} applications, {}μs",
            stat.pass_name,
            stat.applications,
            stat.time_us
        );
    }

    // Example 3: High-performance configuration
    println!("\n--- Example 3: High-Performance Pipeline ---");

    let mut hp_circuit = circuit.clone();
    let hp_compiler = PipelineBuilder::new()
        .with_dead_code_elimination()
        .with_gate_commutation()
        .with_template_substitution()
        .with_gate_fusion()
        .max_iterations(10)
        .min_benefit_score(0.7) // Only run high-benefit passes
        .build();

    let result = hp_compiler.compile(&mut hp_circuit)?;
    println!("High-performance: {} gates → {} gates", circuit.len(), hp_circuit.len());
    println!("Total optimization time: {}μs", result.total_time_us);
    println!("\nFinal circuit:");
    println!("{}", hp_circuit);

    Ok(())
}
