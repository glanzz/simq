//! Demonstration of advanced template pattern matching
//!
//! This example shows how the advanced template matching pass can recognize
//! and optimize complex gate patterns with proper gate replacement.

use simq_compiler::{
    create_compiler, OptimizationLevel,
    passes::{AdvancedTemplateMatching, OptimizationPass},
};
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{Hadamard, PauliX, PauliY, PauliZ};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Template Matching Demo ===\n");

    // Example 1: Hadamard conjugation (H-Z-H = X)
    {
        println!("--- Example 1: H-Z-H → X (Hadamard Conjugation) ---");
        let mut circuit = Circuit::new(1);

        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(PauliZ), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;

        println!("Before: {} gates", circuit.len());
        println!("{}", circuit);

        let pass = AdvancedTemplateMatching::new();
        pass.apply(&mut circuit)?;

        println!("After: {} gates", circuit.len());
        println!("{}", circuit);
        assert_eq!(circuit.len(), 1);
        assert_eq!(circuit.get_operation(0).unwrap().gate().name(), "X");
        println!("✓ Successfully replaced H-Z-H with X\n");
    }

    // Example 2: H-X-H = Z
    {
        println!("--- Example 2: H-X-H → Z (Hadamard Conjugation) ---");
        let mut circuit = Circuit::new(1);

        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;

        println!("Before: {} gates", circuit.len());

        let pass = AdvancedTemplateMatching::new();
        pass.apply(&mut circuit)?;

        println!("After: {} gates", circuit.len());
        assert_eq!(circuit.len(), 1);
        assert_eq!(circuit.get_operation(0).unwrap().gate().name(), "Z");
        println!("✓ Successfully replaced H-X-H with Z\n");
    }

    // Example 3: Self-inverse elimination
    {
        println!("--- Example 3: Self-Inverse Gate Pairs ---");
        let mut circuit = Circuit::new(2);

        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(1)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(1)])?;

        println!("Before: {} gates (X-X and H-H pairs)", circuit.len());

        let pass = AdvancedTemplateMatching::new();
        pass.apply(&mut circuit)?;

        println!("After: {} gates", circuit.len());
        assert_eq!(circuit.len(), 0);
        println!("✓ Successfully removed all self-inverse pairs\n");
    }

    // Example 4: Complex pattern with multiple templates
    {
        println!("--- Example 4: Complex Multi-Template Optimization ---");
        let mut circuit = Circuit::new(2);

        // Pattern 1: H-Z-H on q0 (→ X)
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(PauliZ), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;

        // Pattern 2: X-Y-X on q1 (→ Y)
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(1)])?;
        circuit.add_gate(Arc::new(PauliY), &[QubitId::new(1)])?;
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(1)])?;

        // Pattern 3: Another X-X on q0 (→ I)
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(0)])?;

        println!("Before: {} gates", circuit.len());
        println!("  q0: H-Z-H-X (should become X then cancel with next X)");
        println!("  q1: X-Y-X (should become Y)");

        let pass = AdvancedTemplateMatching::new();
        pass.apply(&mut circuit)?;

        println!("\nAfter: {} gates", circuit.len());
        println!("{}", circuit);

        // H-Z-H → X, then X-X → I, X-Y-X → Y
        assert_eq!(circuit.len(), 1);
        assert_eq!(circuit.get_operation(0).unwrap().gate().name(), "Y");
        println!("✓ Successfully applied multiple template optimizations\n");
    }

    // Example 5: Using with optimization pipeline
    {
        println!("--- Example 5: Full Optimization Pipeline ---");
        let mut circuit = Circuit::new(3);

        // Add a complex pattern
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(PauliZ), &[QubitId::new(0)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(0)])?; // H-Z-H → X

        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(1)])?;
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(1)])?; // X-X → I

        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(2)])?;
        circuit.add_gate(Arc::new(PauliX), &[QubitId::new(2)])?;
        circuit.add_gate(Arc::new(Hadamard), &[QubitId::new(2)])?; // H-X-H → Z
        circuit.add_gate(Arc::new(PauliZ), &[QubitId::new(2)])?;   // Z-Z → I

        println!("Before optimization: {} gates", circuit.len());
        println!("{}", circuit);

        let compiler = create_compiler(OptimizationLevel::O3);
        let result = compiler.compile(&mut circuit)?;

        println!("After O3 optimization: {} gates", circuit.len());
        println!("{}", circuit);
        println!("Optimization time: {}μs", result.total_time_us);
        println!("\n✓ Full pipeline successfully optimized the circuit\n");
    }

    println!("=== All Examples Completed Successfully! ===");
    Ok(())
}
