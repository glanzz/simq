//! Edge case and error path tests for simq-core

use simq_core::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helper: minimal Gate impl usable across all tests
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct MockGate {
    name: String,
    n: usize,
}

impl MockGate {
    fn single(name: &str) -> Arc<dyn Gate> {
        Arc::new(Self {
            name: name.to_string(),
            n: 1,
        })
    }
    fn two(name: &str) -> Arc<dyn Gate> {
        Arc::new(Self {
            name: name.to_string(),
            n: 2,
        })
    }
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }
    fn num_qubits(&self) -> usize {
        self.n
    }
}

// =====================================================================
// Serialization error paths (feature = "serialization")
// =====================================================================

#[cfg(feature = "serialization")]
mod serialization_edge_cases {
    use super::*;
    use simq_gates::standard::{CNot, Hadamard, PauliX};

    #[test]
    fn invalid_json_returns_deserialization_error() {
        let result = Circuit::from_json("not json at all{");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("deserialization") || msg.contains("Deserialization"));
    }

    #[test]
    fn missing_fields_in_json() {
        let json = r#"{"version": 1}"#;
        let result = Circuit::from_json(json);
        assert!(result.is_err());
    }

    #[test]
    fn corrupted_binary_returns_error() {
        let result = Circuit::from_bytes(&[0xff, 0x00, 0xab, 0xcd]);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("deserialization") || msg.contains("Deserialization"));
    }

    #[test]
    fn empty_bytes_returns_error() {
        let result = Circuit::from_bytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn round_trip_binary() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let bytes = circuit.to_bytes().unwrap();
        let restored = Circuit::from_bytes(&bytes).unwrap();
        assert_eq!(restored.num_qubits(), 2);
        assert_eq!(restored.len(), 3);
    }

    #[test]
    fn round_trip_json() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();

        let json = circuit.to_json().unwrap();
        let restored = Circuit::from_json(&json).unwrap();
        assert_eq!(restored.num_qubits(), 3);
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn pretty_json_round_trip() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        let pretty = circuit.to_json_pretty().unwrap();
        assert!(pretty.contains('\n'));
        let restored = Circuit::from_json(&pretty).unwrap();
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn cache_key_differs_for_different_circuits() {
        let mut c1 = Circuit::new(2);
        c1.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();

        let mut c2 = Circuit::new(2);
        c2.add_gate(Arc::new(PauliX), &[QubitId::new(0)]).unwrap();

        assert_ne!(c1.cache_key(), c2.cache_key());
    }

    #[test]
    fn cache_key_same_for_identical_circuits() {
        let build = || {
            let mut c = Circuit::new(2);
            c.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
            c
        };
        assert_eq!(build().cache_key(), build().cache_key());
    }

    #[test]
    fn large_circuit_round_trip() {
        let mut circuit = Circuit::new(10);
        for i in 0..10 {
            circuit
                .add_gate(Arc::new(Hadamard), &[QubitId::new(i)])
                .unwrap();
        }
        let bytes = circuit.to_bytes().unwrap();
        let restored = Circuit::from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 10);
        assert_eq!(restored.num_qubits(), 10);
    }
}

// =====================================================================
// Noise model error paths
// =====================================================================

mod noise_edge_cases {
    use num_complex::Complex64;
    use simq_core::noise::*;

    #[test]
    fn depolarizing_negative_probability() {
        assert!(DepolarizingChannel::new(-0.001).is_err());
    }

    #[test]
    fn depolarizing_above_one() {
        assert!(DepolarizingChannel::new(1.01).is_err());
    }

    #[test]
    fn depolarizing_nan() {
        assert!(DepolarizingChannel::new(f64::NAN).is_err());
    }

    #[test]
    fn depolarizing_boundary_zero() {
        let ch = DepolarizingChannel::new(0.0).unwrap();
        assert!(ch.verify_completeness(1e-10));
    }

    #[test]
    fn depolarizing_boundary_one() {
        let ch = DepolarizingChannel::new(1.0).unwrap();
        assert!(ch.verify_completeness(1e-10));
    }

    #[test]
    fn amplitude_damping_negative() {
        assert!(AmplitudeDamping::new(-0.01).is_err());
    }

    #[test]
    fn amplitude_damping_above_one() {
        assert!(AmplitudeDamping::new(1.1).is_err());
    }

    #[test]
    fn amplitude_damping_nan() {
        assert!(AmplitudeDamping::new(f64::NAN).is_err());
    }

    #[test]
    fn amplitude_damping_from_t1_negative_t1() {
        assert!(AmplitudeDamping::from_t1(-1.0, 0.1).is_err());
    }

    #[test]
    fn amplitude_damping_from_t1_negative_gate_time() {
        assert!(AmplitudeDamping::from_t1(50.0, -0.1).is_err());
    }

    #[test]
    fn amplitude_damping_from_t1_zero_gate_time() {
        let ch = AmplitudeDamping::from_t1(50.0, 0.0).unwrap();
        assert!((ch.gamma() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn phase_damping_above_half() {
        assert!(PhaseDamping::new(0.51).is_err());
    }

    #[test]
    fn phase_damping_negative() {
        assert!(PhaseDamping::new(-0.01).is_err());
    }

    #[test]
    fn phase_damping_nan() {
        assert!(PhaseDamping::new(f64::NAN).is_err());
    }

    #[test]
    fn phase_damping_boundary_half() {
        let ch = PhaseDamping::new(0.5).unwrap();
        assert!(ch.verify_completeness(1e-10));
    }

    #[test]
    fn phase_damping_from_t2_negative() {
        assert!(PhaseDamping::from_t2(-1.0, 0.1).is_err());
    }

    #[test]
    fn readout_error_negative_p01() {
        assert!(ReadoutError::new(-0.01, 0.5).is_err());
    }

    #[test]
    fn readout_error_p10_above_one() {
        assert!(ReadoutError::new(0.5, 1.01).is_err());
    }

    #[test]
    fn noise_model_composition() {
        let mut model = NoiseModel::new();
        assert!(!model.has_noise());

        model.set_gate_noise(DepolarizingChannel::new(0.01).unwrap());
        assert!(model.has_noise());

        model.set_readout_noise(ReadoutError::symmetric(0.02).unwrap());
        model.set_idle_noise(PhaseDamping::new(0.01).unwrap());
        assert!(model.has_noise());
    }

    #[test]
    fn channel_description() {
        let ch = DepolarizingChannel::new(0.05).unwrap();
        let desc = ch.description();
        assert!(desc.contains("depolarizing"));
    }

    #[test]
    fn kraus_operator_invalid_dimension() {
        use simq_core::noise::KrausOperator;
        let result = KrausOperator::new(vec![Complex64::new(1.0, 0.0); 9], 3);
        assert!(result.is_err());
    }

    #[test]
    fn kraus_operator_size_mismatch() {
        use simq_core::noise::KrausOperator;
        let result = KrausOperator::new(vec![Complex64::new(1.0, 0.0); 3], 2);
        assert!(result.is_err());
    }
}

// =====================================================================
// Validation rules with boundary inputs
// =====================================================================

mod validation_edge_cases {
    use super::*;
    use simq_core::validation::dag::DependencyGraph;
    use simq_core::validation::rules::*;

    #[test]
    fn qubit_usage_rule_valid_circuit() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(2)])
            .unwrap();

        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let rule = QubitUsageRule;
        let result = rule.validate(&circuit, &dag);
        assert!(result.is_valid());
    }

    #[test]
    fn cycle_detection_simple_circuit() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(0)])
            .unwrap();

        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let rule = CycleDetectionRule;
        let result = rule.validate(&circuit, &dag);
        assert!(result.is_valid());
    }

    #[test]
    fn dependency_validation_with_two_qubit_gate() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::two("CNOT"), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(1)])
            .unwrap();

        let dag = DependencyGraph::from_circuit(&circuit).unwrap();
        let rule = DependencyValidationRule;
        let result = rule.validate(&circuit, &dag);
        assert!(result.is_valid());
    }

    #[test]
    fn validate_dag_on_empty_circuit() {
        let circuit = Circuit::new(1);
        let report = circuit.validate_dag().unwrap();
        assert!(!report.has_errors());
    }

    #[test]
    fn validate_dag_multi_qubit() {
        let mut circuit = Circuit::new(4);
        for i in 0..4 {
            circuit
                .add_gate(MockGate::single("H"), &[QubitId::new(i)])
                .unwrap();
        }
        circuit
            .add_gate(MockGate::two("CX"), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(MockGate::two("CX"), &[QubitId::new(2), QubitId::new(3)])
            .unwrap();

        let report = circuit.validate_dag().unwrap();
        assert!(!report.has_errors());
    }
}

// =====================================================================
// Circuit debugger edge cases
// =====================================================================

mod debugger_edge_cases {
    use super::*;

    #[test]
    fn debugger_on_empty_circuit() {
        let circuit = Circuit::new(2);
        let mut debugger = CircuitDebugger::new(&circuit);

        assert_eq!(debugger.total_gates(), 0);
        assert!(debugger.is_at_start());
        assert!(debugger.is_at_end());
        assert!(!debugger.has_next());
        assert!(!debugger.step());
        assert!(!debugger.step_back());
    }

    #[test]
    fn debugger_breakpoint_at_out_of_bounds() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        debugger.add_breakpoint(100);
        assert!(debugger.breakpoints().is_empty());
    }

    #[test]
    fn debugger_duplicate_breakpoint() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(0)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        debugger.add_breakpoint(0);
        debugger.add_breakpoint(0);
        assert_eq!(debugger.breakpoints().len(), 1);
    }

    #[test]
    fn debugger_jump_to_end() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(1)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        assert!(debugger.jump_to(2));
        assert!(debugger.is_at_end());
        assert_eq!(debugger.history().len(), 2);
    }

    #[test]
    fn debugger_jump_backward_resets_history() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("Z"), &[QubitId::new(0)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        debugger.jump_to(3);
        assert_eq!(debugger.history().len(), 3);

        debugger.jump_to(1);
        assert_eq!(debugger.step_number(), 1);
        assert_eq!(debugger.history().len(), 1);
    }

    #[test]
    fn debugger_continue_without_breakpoints() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        let hit = debugger.continue_execution();
        assert!(!hit);
        assert!(debugger.is_at_end());
    }

    #[test]
    fn debugger_status_display() {
        let circuit = Circuit::new(2);
        let debugger = CircuitDebugger::new(&circuit);
        let status = debugger.status();
        let display = format!("{}", status);
        assert!(display.contains("Debugger Status"));
    }

    #[test]
    fn debugger_step_info_display() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        debugger.step();

        let info = &debugger.history()[0];
        let display = format!("{}", info);
        assert!(display.contains("H"));
    }

    #[test]
    fn debugger_visualize_at_start() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();

        let debugger = CircuitDebugger::new(&circuit);
        let viz = debugger.visualize_current_position();
        assert!(viz.contains("Step 0/1"));
    }

    #[test]
    fn debugger_executed_and_remaining() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("Z"), &[QubitId::new(0)])
            .unwrap();

        let mut debugger = CircuitDebugger::new(&circuit);
        debugger.step();
        assert_eq!(debugger.executed_operations().len(), 1);
        assert_eq!(debugger.remaining_operations().len(), 2);
    }
}

// =====================================================================
// Error type Display and formatting
// =====================================================================

mod error_display {
    use super::*;

    #[test]
    fn all_error_variants_display() {
        let errors: Vec<QuantumError> = vec![
            QuantumError::InvalidQubit(5, 3),
            QuantumError::InvalidQubitCount {
                gate: "CNOT".into(),
                expected: 2,
                actual: 1,
            },
            QuantumError::EmptyCircuit,
            QuantumError::DuplicateQubit(QubitId::new(0)),
            QuantumError::ValidationError("test".into()),
            QuantumError::SerializationError("test".into()),
            QuantumError::DeserializationError("test".into()),
            QuantumError::UnknownGateType("FooGate".into()),
            QuantumError::VersionMismatch {
                expected: 1,
                actual: 99,
            },
            QuantumError::CacheError("test".into()),
            QuantumError::CycleDetected {
                operations: vec![0, 1, 2],
            },
            QuantumError::InvalidDependency {
                from: 0,
                to: 1,
                qubit: 0,
            },
            QuantumError::TopologicalOrderError {
                reason: "test".into(),
            },
        ];

        for err in &errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty(), "Error display should not be empty: {:?}", err);
        }
    }

    #[test]
    fn error_is_std_error() {
        let err = QuantumError::EmptyCircuit;
        let _: &dyn std::error::Error = &err;
    }
}

// =====================================================================
// Visualization edge cases
// =====================================================================

mod visualization_edge_cases {
    use super::*;

    #[test]
    fn ascii_render_empty_circuit() {
        let circuit = Circuit::new(2);
        let rendered = circuit.to_ascii();
        assert!(!rendered.is_empty());
    }

    #[test]
    fn ascii_render_single_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        let rendered = circuit.to_ascii();
        assert!(rendered.contains("H"));
    }

    #[test]
    fn ascii_render_with_custom_config() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();

        let config = AsciiConfig {
            max_width: 40,
            compact: true,
            ..Default::default()
        };
        let rendered = circuit.to_ascii_with_config(&config);
        assert!(!rendered.is_empty());
    }

    #[test]
    fn latex_render_empty_circuit() {
        let circuit = Circuit::new(2);
        let rendered = circuit.to_latex();
        assert!(!rendered.is_empty());
    }

    #[test]
    fn latex_render_single_gate() {
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        let rendered = circuit.to_latex();
        assert!(!rendered.is_empty());
    }

    #[test]
    fn circuit_display_format() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        let display = format!("{}", circuit);
        assert!(display.contains("3 qubits"));
        assert!(display.contains("1 operations"));
    }

    #[test]
    fn deep_circuit_rendering() {
        let mut circuit = Circuit::new(2);
        for _ in 0..20 {
            circuit
                .add_gate(MockGate::single("X"), &[QubitId::new(0)])
                .unwrap();
            circuit
                .add_gate(MockGate::single("Y"), &[QubitId::new(1)])
                .unwrap();
        }
        let rendered = circuit.to_ascii();
        assert!(!rendered.is_empty());
    }
}

// =====================================================================
// Circuit operations edge cases
// =====================================================================

mod circuit_ops_edge_cases {
    use super::*;

    #[test]
    fn append_mismatched_qubits() {
        let c1 = Circuit::new(2);
        let c2 = Circuit::new(3);
        let mut c1_mut = c1;
        let result = c1_mut.append(&c2);
        assert!(result.is_err());
    }

    #[test]
    fn remove_operation_out_of_bounds() {
        let mut circuit = Circuit::new(2);
        assert!(circuit.remove_operation(0).is_none());
        assert!(circuit.remove_operation(100).is_none());
    }

    #[test]
    fn insert_operation_out_of_bounds() {
        let circuit = Circuit::new(2);
        let gate = MockGate::single("H");
        let op = GateOp::new(gate, &[QubitId::new(0)]).unwrap();
        let mut circuit = circuit;
        let result = circuit.insert_operation(100, op);
        assert!(result.is_err());
    }

    #[test]
    fn gate_op_duplicate_qubits() {
        let gate = MockGate::two("CNOT");
        let result = GateOp::new(gate, &[QubitId::new(0), QubitId::new(0)]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), QuantumError::DuplicateQubit(_)));
    }

    #[test]
    fn circuit_depth_with_parallel_gates() {
        let mut circuit = Circuit::new(4);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(2)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(3)])
            .unwrap();
        assert_eq!(circuit.depth(), 1);
    }

    #[test]
    fn circuit_gate_counts() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("H"), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(MockGate::single("X"), &[QubitId::new(0)])
            .unwrap();

        let counts = circuit.gate_counts();
        assert_eq!(counts.get("H"), Some(&2));
        assert_eq!(counts.get("X"), Some(&1));
    }
}
