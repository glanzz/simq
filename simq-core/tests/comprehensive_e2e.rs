//! Comprehensive end-to-end tests for simq-core
//!
//! Covers: CircuitBuilder, DynamicCircuitBuilder, Circuit, Gate, GateOp,
//! Parameters, Visualization, Debugging, Noise, Validation, and Serialization.

use simq_core::*;
use simq_gates::standard::*;
use std::sync::Arc;

// ============================================================================
// 1. CircuitBuilder (compile-time sized)
// ============================================================================

#[test]
fn build_empty_circuit() {
    let builder = CircuitBuilder::<3>::new();
    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 3);
    assert_eq!(circuit.len(), 0);
    assert!(circuit.is_empty());
    assert!(circuit.validate().is_ok());
}

#[test]
fn build_single_gate_circuit() {
    let mut builder = CircuitBuilder::<2>::new();
    let [q0, _q1] = builder.qubits();
    builder.apply_gate(Arc::new(Hadamard), &[q0]).unwrap();
    let circuit = builder.build();
    assert_eq!(circuit.len(), 1);
    assert_eq!(circuit.get_operation(0).unwrap().gate().name(), "H");
}

#[test]
fn build_multi_qubit_circuits() {
    for n in [1usize, 2, 3, 5] {
        let mut builder = DynamicCircuitBuilder::new(n);
        builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
        let circuit = builder.build();
        assert_eq!(circuit.num_qubits(), n);
    }
}

#[test]
fn apply_single_qubit_gates() {
    let mut builder = CircuitBuilder::<1>::new();
    let [q0] = builder.qubits();
    let gates: Vec<Arc<dyn Gate>> = vec![
        Arc::new(Hadamard),
        Arc::new(PauliX),
        Arc::new(PauliY),
        Arc::new(PauliZ),
        Arc::new(SGate),
        Arc::new(TGate),
        Arc::new(SXGate),
        Arc::new(Identity),
    ];
    for g in &gates {
        builder.apply_gate(g.clone(), &[q0]).unwrap();
    }
    let circuit = builder.build();
    assert_eq!(circuit.len(), gates.len());
    assert_eq!(circuit.single_qubit_gate_count(), gates.len());
    assert!(circuit.validate().is_ok());
}

#[test]
fn apply_two_qubit_gates() {
    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, _q2] = builder.qubits();
    let gates: Vec<Arc<dyn Gate>> = vec![
        Arc::new(CNot),
        Arc::new(CZ),
        Arc::new(Swap),
        Arc::new(ISwap),
        Arc::new(CY),
        Arc::new(ECR),
    ];
    for g in &gates {
        builder.apply_gate(g.clone(), &[q0, q1]).unwrap();
    }
    let circuit = builder.build();
    assert_eq!(circuit.two_qubit_gate_count(), gates.len());
    assert!(circuit.validate().is_ok());
}

#[test]
fn apply_three_qubit_gates() {
    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = builder.qubits();
    builder
        .apply_gate(Arc::new(Toffoli), &[q0, q1, q2])
        .unwrap();
    builder
        .apply_gate(Arc::new(Fredkin), &[q0, q1, q2])
        .unwrap();
    let circuit = builder.build();
    assert_eq!(circuit.len(), 2);
    assert!(circuit.validate().is_ok());
}

#[test]
fn apply_rotation_gates() {
    let mut builder = CircuitBuilder::<1>::new();
    let [q0] = builder.qubits();
    let angles = [
        0.0,
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::PI,
        std::f64::consts::TAU,
    ];
    for &theta in &angles {
        builder
            .apply_gate(Arc::new(RotationX::new(theta)), &[q0])
            .unwrap();
        builder
            .apply_gate(Arc::new(RotationY::new(theta)), &[q0])
            .unwrap();
        builder
            .apply_gate(Arc::new(RotationZ::new(theta)), &[q0])
            .unwrap();
        builder
            .apply_gate(Arc::new(Phase::new(theta)), &[q0])
            .unwrap();
    }
    let circuit = builder.build();
    assert_eq!(circuit.len(), angles.len() * 4);
    assert!(circuit.validate().is_ok());
}

#[test]
fn qubit_index_bounds() {
    let mut builder = DynamicCircuitBuilder::new(2);
    let result = builder.apply_gate(Arc::new(Hadamard), &[5]);
    assert!(result.is_err());
}

#[test]
fn duplicate_qubit_in_gate() {
    let mut builder = DynamicCircuitBuilder::new(3);
    let result = builder.apply_gate(Arc::new(CNot), &[0, 0]);
    assert!(result.is_err());
}

#[test]
fn large_circuit_100_gates() {
    let mut builder = DynamicCircuitBuilder::new(5);
    for i in 0..100 {
        builder.apply_gate(Arc::new(Hadamard), &[i % 5]).unwrap();
    }
    let circuit = builder.build();
    assert_eq!(circuit.len(), 100);
    assert!(circuit.validate().is_ok());
}

#[test]
fn circuit_gate_count() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(Hadamard), &[1]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[2]).unwrap();
    let circuit = builder.build();
    let counts = circuit.gate_counts();
    assert_eq!(*counts.get("H").unwrap(), 2);
    assert_eq!(*counts.get("X").unwrap(), 1);
}

#[test]
fn circuit_num_qubits() {
    let circuit = Circuit::new(7);
    assert_eq!(circuit.num_qubits(), 7);
}

#[test]
fn circuit_gate_ordering() {
    let mut builder = DynamicCircuitBuilder::new(3);
    let gate_names = ["H", "X", "Y", "Z"];
    let gates: Vec<Arc<dyn Gate>> = vec![
        Arc::new(Hadamard),
        Arc::new(PauliX),
        Arc::new(PauliY),
        Arc::new(PauliZ),
    ];
    for (i, g) in gates.iter().enumerate() {
        builder.apply_gate(g.clone(), &[i % 3]).unwrap();
    }
    let circuit = builder.build();
    for (i, op) in circuit.operations().enumerate() {
        assert_eq!(op.gate().name(), gate_names[i]);
    }
}

// ============================================================================
// 2. DynamicCircuitBuilder (runtime-sized)
// ============================================================================

#[test]
fn dynamic_build_n_qubits() {
    for n in [1, 2, 5, 10, 20] {
        let builder = DynamicCircuitBuilder::new(n);
        assert_eq!(builder.num_qubits(), n);
        let circuit = builder.build();
        assert_eq!(circuit.num_qubits(), n);
    }
}

#[test]
fn dynamic_qubit_out_of_range() {
    let mut builder = DynamicCircuitBuilder::new(3);
    assert!(builder.apply_gate(Arc::new(Hadamard), &[3]).is_err());
    assert!(builder.apply_gate(Arc::new(Hadamard), &[100]).is_err());
}

#[test]
fn dynamic_matches_static() {
    let mut static_builder = CircuitBuilder::<3>::new();
    let [q0, q1, _q2] = static_builder.qubits();
    static_builder
        .apply_gate(Arc::new(Hadamard), &[q0])
        .unwrap();
    static_builder
        .apply_gate(Arc::new(CNot), &[q0, q1])
        .unwrap();
    let static_circuit = static_builder.build();

    let mut dynamic_builder = DynamicCircuitBuilder::new(3);
    dynamic_builder
        .apply_gate(Arc::new(Hadamard), &[0])
        .unwrap();
    dynamic_builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let dynamic_circuit = dynamic_builder.build();

    assert_eq!(static_circuit.num_qubits(), dynamic_circuit.num_qubits());
    assert_eq!(static_circuit.len(), dynamic_circuit.len());
    for (s_op, d_op) in static_circuit
        .operations()
        .zip(dynamic_circuit.operations())
    {
        assert_eq!(s_op.gate().name(), d_op.gate().name());
        assert_eq!(s_op.qubits(), d_op.qubits());
    }
}

#[test]
#[should_panic(expected = "at least one qubit")]
fn dynamic_zero_qubits() {
    DynamicCircuitBuilder::new(0);
}

#[test]
fn dynamic_large_qubit_count() {
    let builder = DynamicCircuitBuilder::new(25);
    assert_eq!(builder.num_qubits(), 25);
    let circuit = builder.build();
    assert_eq!(circuit.num_qubits(), 25);
}

// ============================================================================
// 3. Parameters & Parametric Circuits
// ============================================================================

#[test]
fn create_parameter() {
    let param = Parameter::named("theta", 0.0);
    assert_eq!(param.name(), Some("theta"));
    assert_eq!(param.value(), 0.0);
}

#[test]
fn parameter_registry_add_get() {
    let mut registry = ParameterRegistry::new();
    let id = registry.add_named("theta", 0.5);
    let param = registry.get(id).unwrap();
    assert_eq!(param.name(), Some("theta"));
    assert_eq!(param.value(), 0.5);
}

#[test]
fn parameter_registry_duplicates() {
    let mut registry = ParameterRegistry::new();
    let id1 = registry.add_named("theta", 1.0);
    let id2 = registry.add_named("theta", 2.0);
    assert_ne!(id1, id2);
}

#[test]
fn bind_parameter_value() {
    let mut param = Parameter::named("theta", 0.0);
    param.set_value(1.5).unwrap();
    assert_eq!(param.value(), 1.5);
}

#[test]
fn parameter_bounds() {
    let param = Parameter::new(0.5)
        .with_bounds(0.0, std::f64::consts::TAU)
        .unwrap();
    let bounds = param.bounds();
    assert!(bounds.is_some());
    let (lo, hi) = bounds.unwrap();
    assert_eq!(lo, 0.0);
    assert_eq!(hi, std::f64::consts::TAU);
}

#[test]
fn parameter_bounds_enforcement() {
    let mut param = Parameter::new(0.5).with_bounds(0.0, 1.0).unwrap();
    assert!(param.set_value(0.5).is_ok());
    assert!(param.set_value(2.0).is_err());
    assert!(param.set_value(-0.1).is_err());
}

#[test]
fn parameter_freeze_unfreeze() {
    let param = Parameter::named("theta", 1.0);
    let frozen = param.as_frozen();
    assert!(frozen.is_frozen());
    let mut unfrozen = frozen;
    unfrozen.unfreeze();
    assert!(!unfrozen.is_frozen());
}

#[test]
fn parameter_registry_workflow() {
    let mut registry = ParameterRegistry::new();
    let ids = registry.add_many(&[1.0, 2.0, 3.0]);
    assert_eq!(ids.len(), 3);
    let values = vec![10.0, 20.0, 30.0];
    registry.set_values(&ids, &values).unwrap();
    let retrieved = registry.get_values(&ids);
    assert_eq!(retrieved, vec![10.0, 20.0, 30.0]);
}

#[test]
fn parameter_registry_set_all_values() {
    let mut registry = ParameterRegistry::new();
    let _ids = registry.add_many(&[0.0, 0.0, 0.0]);
    registry.set_all_values(&[10.0, 20.0, 30.0]).unwrap();
    let all = registry.all_values();
    assert_eq!(all, vec![10.0, 20.0, 30.0]);
}

#[test]
fn parameter_registry_named_lookup() {
    let mut registry = ParameterRegistry::new();
    registry.add_named("alpha", 1.0);
    registry.add_named("beta", 2.0);
    let alpha = registry.get_by_name("alpha").unwrap();
    assert_eq!(alpha.value(), 1.0);
    let beta_id = registry.get_id_by_name("beta").unwrap();
    assert_eq!(registry.get(beta_id).unwrap().value(), 2.0);
}

#[test]
fn parameter_registry_frozen_unfrozen() {
    let mut registry = ParameterRegistry::new();
    let id1 = registry.add_named("free", 1.0);
    let id2 = registry.add_named("frozen", 2.0);
    registry.get_mut(id2).unwrap().freeze();
    let frozen = registry.frozen_params();
    let unfrozen = registry.unfrozen_params();
    assert_eq!(frozen.len(), 1);
    assert_eq!(unfrozen.len(), 1);
    assert_eq!(unfrozen[0], id1);
    assert_eq!(frozen[0], id2);
}

// ============================================================================
// 4. Circuit Visualization
// ============================================================================

#[test]
fn ascii_render_basic() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let circuit = builder.build();
    let ascii = circuit.to_ascii();
    assert!(!ascii.is_empty());
    assert!(ascii.contains("H"));
}

#[test]
fn ascii_render_multi_qubit() {
    let mut builder = DynamicCircuitBuilder::new(4);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[2, 3]).unwrap();
    let circuit = builder.build();
    let ascii = circuit.to_ascii();
    assert!(!ascii.is_empty());
    assert!(ascii.contains("q0"));
}

#[test]
fn ascii_render_empty() {
    let circuit = Circuit::new(2);
    let ascii = circuit.to_ascii();
    assert!(!ascii.is_empty());
}

#[test]
fn latex_render_basic() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();
    let latex = circuit.to_latex();
    assert!(!latex.is_empty());
}

#[test]
fn latex_render_all_gate_types() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    builder.apply_gate(Arc::new(PauliZ), &[2]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder
        .apply_gate(Arc::new(RotationX::new(1.0)), &[0])
        .unwrap();
    let circuit = builder.build();
    let latex = circuit.to_latex();
    assert!(!latex.is_empty());
}

#[test]
fn bloch_sphere_known_states() {
    use num_complex::Complex64;

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    // |0⟩ → north pole (0, 0, 1)
    let zero = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let bv = BlochVector::from_state(&zero);
    assert!((bv.z - 1.0).abs() < 1e-10);
    assert!(bv.x.abs() < 1e-10);
    assert!(bv.y.abs() < 1e-10);

    // |1⟩ → south pole (0, 0, -1)
    let one = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let bv = BlochVector::from_state(&one);
    assert!((bv.z + 1.0).abs() < 1e-10);

    // |+⟩ → (1, 0, 0)
    let plus = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
    ];
    let bv = BlochVector::from_state(&plus);
    assert!((bv.x - 1.0).abs() < 1e-10);
    assert!(bv.y.abs() < 1e-10);
    assert!(bv.z.abs() < 1e-10);

    // |−⟩ → (-1, 0, 0)
    let minus = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(-inv_sqrt2, 0.0),
    ];
    let bv = BlochVector::from_state(&minus);
    assert!((bv.x + 1.0).abs() < 1e-10);

    // |i⟩ = (|0⟩ + i|1⟩)/√2 → (0, 1, 0)
    let i_state = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(0.0, inv_sqrt2),
    ];
    let bv = BlochVector::from_state(&i_state);
    assert!(bv.x.abs() < 1e-10);
    assert!((bv.y - 1.0).abs() < 1e-10);
    assert!(bv.z.abs() < 1e-10);
}

// ============================================================================
// 5. Circuit Debugging
// ============================================================================

#[test]
fn debugger_step_through() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);
    assert_eq!(debugger.step_number(), 0);
    assert!(debugger.is_at_start());

    assert!(debugger.step());
    assert_eq!(debugger.history().last().unwrap().gate_name, "H");

    assert!(debugger.step());
    assert_eq!(debugger.history().last().unwrap().gate_name, "X");

    assert!(debugger.step());
    assert_eq!(debugger.history().last().unwrap().gate_name, "CNOT");

    assert!(!debugger.step());
    assert!(debugger.is_at_end());
}

#[test]
fn debugger_step_back() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);
    debugger.step();
    debugger.step();
    assert!(debugger.step_back());
    assert!(debugger.step());
    assert_eq!(debugger.history().last().unwrap().gate_name, "X");
}

#[test]
fn debugger_breakpoints() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliY), &[1]).unwrap();
    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);
    debugger.add_breakpoint(2);
    let hit_breakpoint = debugger.continue_execution();
    assert!(hit_breakpoint);
    assert_eq!(debugger.step_number(), 2);
}

#[test]
fn debugger_empty_circuit() {
    let circuit = Circuit::new(2);
    let mut debugger = CircuitDebugger::new(&circuit);
    assert!(!debugger.step());
    assert!(debugger.is_at_end());
}

#[test]
fn debugger_reset() {
    let mut builder = DynamicCircuitBuilder::new(1);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);
    debugger.step();
    assert!(debugger.is_at_end());
    debugger.reset();
    assert!(debugger.is_at_start());
}

#[test]
fn debugger_history() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);
    debugger.step();
    debugger.step();
    let history = debugger.history();
    assert_eq!(history.len(), 2);
}

#[test]
fn debugger_jump_to() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliY), &[1]).unwrap();
    let circuit = builder.build();

    let mut debugger = CircuitDebugger::new(&circuit);
    debugger.jump_to(2);
    let executed = debugger.executed_operations();
    assert_eq!(executed.len(), 2);
}

// ============================================================================
// 6. Noise Models
// ============================================================================

#[test]
fn depolarizing_channel_identity() {
    let channel = DepolarizingChannel::new(0.0).unwrap();
    let kraus = channel.kraus_operators();
    assert!(!kraus.is_empty());
}

#[test]
fn depolarizing_channel_full() {
    let channel = DepolarizingChannel::new(1.0).unwrap();
    let kraus = channel.kraus_operators();
    assert!(!kraus.is_empty());
}

#[test]
fn depolarizing_invalid_probability() {
    assert!(DepolarizingChannel::new(-0.1).is_err());
    assert!(DepolarizingChannel::new(1.1).is_err());
}

#[test]
fn amplitude_damping_decay() {
    let channel = AmplitudeDamping::new(0.5).unwrap();
    let kraus = channel.kraus_operators();
    assert!(!kraus.is_empty());
    assert!(channel.gamma() > 0.0);
}

#[test]
fn phase_damping_valid() {
    let channel = PhaseDamping::new(0.3).unwrap();
    let kraus = channel.kraus_operators();
    assert!(!kraus.is_empty());
}

#[test]
fn readout_error_creation() {
    let re = ReadoutError::new(0.1, 0.2).unwrap();
    assert_eq!(re.p01(), 0.1);
    assert_eq!(re.p10(), 0.2);
    let re_sym = ReadoutError::symmetric(0.05).unwrap();
    assert_eq!(re_sym.p01(), 0.05);
    assert_eq!(re_sym.p10(), 0.05);
}

#[test]
fn readout_error_invalid() {
    assert!(ReadoutError::new(-0.1, 0.2).is_err());
    assert!(ReadoutError::new(0.1, 1.5).is_err());
}

#[test]
fn hardware_noise_model_creation() {
    let model = HardwareNoiseModel::new(5);
    assert_eq!(model.num_qubits(), 5);
    assert!(model.qubit(0).is_some());
    assert!(model.qubit(5).is_none());
}

#[test]
fn hardware_noise_model_presets() {
    let ibm = HardwareNoiseModel::ibm_washington();
    assert_eq!(ibm.num_qubits(), 127);

    let google = HardwareNoiseModel::google_sycamore();
    assert_eq!(google.num_qubits(), 53);

    let ionq = HardwareNoiseModel::ionq_aria();
    assert_eq!(ionq.num_qubits(), 25);

    let falcon = HardwareNoiseModel::ibm_falcon_5q();
    assert_eq!(falcon.num_qubits(), 5);
}

#[test]
fn hardware_noise_qubit_properties() {
    let mut model = HardwareNoiseModel::new(3);
    model.set_qubit_t1(0, 150.0);
    model.set_qubit_t2(0, 120.0);
    model.set_readout_error(0, 0.01, 0.015);
    model.set_single_qubit_fidelity(0, 0.9999);

    let props = model.qubit(0).unwrap();
    assert_eq!(props.t1, 150.0);
    assert_eq!(props.t2, 120.0);
    assert_eq!(props.readout_p01, 0.01);
    assert_eq!(props.readout_p10, 0.015);
    assert_eq!(props.single_qubit_gate_fidelity, 0.9999);
}

#[test]
fn hardware_noise_gate_timing() {
    let mut model = HardwareNoiseModel::new(2);
    let timing = GateTiming {
        single_qubit_gate_time: 0.03,
        two_qubit_gate_time: 0.2,
        measurement_time: 1.5,
    };
    model.set_timing(timing);
    let t = model.timing();
    assert_eq!(t.single_qubit_gate_time, 0.03);
    assert_eq!(t.two_qubit_gate_time, 0.2);
    assert_eq!(t.measurement_time, 1.5);
}

#[test]
fn hardware_noise_two_qubit_gate() {
    let mut model = HardwareNoiseModel::new(3);
    model.set_two_qubit_gate(0, 1, 0.99, 0.3);
    let props = model.two_qubit_gate_properties(0, 1).unwrap();
    assert_eq!(props.fidelity, 0.99);
    assert_eq!(props.duration, 0.3);
}

#[test]
fn hardware_noise_idle_noise() {
    let model = HardwareNoiseModel::ibm_falcon_5q();
    let (amp, phase) = model.idle_noise(0, 5.0).unwrap();
    assert!(amp.gamma() > 0.0);
    assert!(phase.lambda() > 0.0);
}

#[test]
fn hardware_noise_circuit_fidelity() {
    let model = HardwareNoiseModel::ibm_falcon_5q();
    let single_qubit_gates = vec![2, 1, 0, 0, 0]; // 2 gates on q0, 1 on q1
    let two_qubit_gates = vec![(0, 1)];
    let fidelity = model.estimate_circuit_fidelity(&single_qubit_gates, &two_qubit_gates, 1.0);
    assert!(fidelity > 0.0 && fidelity <= 1.0);
}

#[test]
fn hardware_noise_crosstalk() {
    let mut model = HardwareNoiseModel::new(3);
    model.set_crosstalk(0, 1, 0.01, 0.001);
    let props = model.crosstalk_properties(0, 1).unwrap();
    assert_eq!(props.zz_coupling, 0.01);
    assert_eq!(props.spectator_error, 0.001);
}

#[test]
fn hardware_noise_single_gate_noise() {
    let model = HardwareNoiseModel::new(2);
    let noise = model.single_qubit_gate_noise(0).unwrap();
    assert!(!noise.amplitude_damping.is_empty());
    assert!(!noise.phase_damping.is_empty());
    assert!(!noise.depolarizing.is_empty());
    assert_eq!(noise.qubits, vec![0]);
}

#[test]
fn hardware_noise_two_qubit_gate_noise() {
    let model = HardwareNoiseModel::new(3);
    let noise = model.two_qubit_gate_noise(0, 1).unwrap();
    assert_eq!(noise.amplitude_damping.len(), 2);
    assert_eq!(noise.phase_damping.len(), 2);
    assert_eq!(noise.qubits, vec![0, 1]);
}

#[test]
fn hardware_noise_readout_error() {
    let mut model = HardwareNoiseModel::new(2);
    model.set_readout_error(0, 0.02, 0.03);
    let readout = model.readout_error(0).unwrap();
    assert_eq!(readout.p01(), 0.02);
    assert_eq!(readout.p10(), 0.03);
}

#[test]
fn hardware_noise_enable_disable() {
    let mut model = HardwareNoiseModel::new(2);
    model.set_idle_noise_enabled(false);
    assert!(!model.is_idle_noise_enabled());
    model.set_idle_noise_enabled(true);
    assert!(model.is_idle_noise_enabled());
    model.set_crosstalk_enabled(true);
    assert!(model.is_crosstalk_enabled());
    model.set_crosstalk_enabled(false);
    assert!(!model.is_crosstalk_enabled());
}

// ============================================================================
// 7. Validation
// ============================================================================

#[test]
fn validate_well_formed_circuit() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[2]).unwrap();
    let circuit = builder.build();
    assert!(circuit.validate().is_ok());
    assert!(circuit.validate_dag().is_ok());
}

#[test]
fn validate_dag_acyclicity() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let circuit = builder.build();
    assert!(circuit.is_acyclic().unwrap());
}

#[test]
fn validate_dag_dependencies() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    let circuit = builder.build();
    assert!(circuit.validate_dag().is_ok());
    assert_eq!(circuit.compute_depth().unwrap(), 2);
}

#[test]
fn validation_report_format() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();
    let report = circuit.validate_dag().unwrap();
    let formatted = report.format(&circuit);
    assert!(!formatted.is_empty());
}

#[test]
fn validate_circuit_depth_parallel() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[1]).unwrap();
    builder.apply_gate(Arc::new(PauliY), &[2]).unwrap();
    let circuit = builder.build();
    assert_eq!(circuit.compute_depth().unwrap(), 1);
}

#[test]
fn validate_circuit_depth_sequential() {
    let mut builder = DynamicCircuitBuilder::new(1);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliY), &[0]).unwrap();
    let circuit = builder.build();
    assert_eq!(circuit.compute_depth().unwrap(), 3);
}

#[test]
fn validate_parallelism_analysis() {
    let mut builder = DynamicCircuitBuilder::new(4);
    for q in 0..4 {
        builder.apply_gate(Arc::new(Hadamard), &[q]).unwrap();
    }
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[2, 3]).unwrap();
    let circuit = builder.build();
    let analysis = circuit.analyze_parallelism().unwrap();
    assert!(analysis.parallelism_factor > 1.0);
}

#[test]
fn validate_large_circuit_performance() {
    use std::time::Instant;
    let mut builder = DynamicCircuitBuilder::new(10);
    for i in 0..1000 {
        builder.apply_gate(Arc::new(Hadamard), &[i % 10]).unwrap();
    }
    let circuit = builder.build();
    let start = Instant::now();
    assert!(circuit.validate().is_ok());
    let elapsed = start.elapsed();
    assert!(elapsed.as_millis() < 500, "Validation took {:?}", elapsed);
}

// ============================================================================
// 8. Circuit Operations
// ============================================================================

#[test]
fn circuit_append() {
    let mut c1 = Circuit::new(2);
    c1.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
    let mut c2 = Circuit::new(2);
    c2.add_gate(Arc::new(PauliX), &[QubitId::new(1)]).unwrap();
    c1.append(&c2).unwrap();
    assert_eq!(c1.len(), 2);
    assert_eq!(c1.get_operation(0).unwrap().gate().name(), "H");
    assert_eq!(c1.get_operation(1).unwrap().gate().name(), "X");
}

#[test]
fn circuit_append_mismatch() {
    let mut c1 = Circuit::new(2);
    let c2 = Circuit::new(3);
    assert!(c1.append(&c2).is_err());
}

#[test]
fn circuit_remove_operation() {
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
        .unwrap();
    let removed = circuit.remove_operation(0);
    assert!(removed.is_some());
    assert_eq!(removed.unwrap().gate().name(), "H");
    assert_eq!(circuit.len(), 1);
}

#[test]
fn circuit_insert_operation() {
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliZ), &[QubitId::new(0)])
        .unwrap();
    let x_op = GateOp::new(Arc::new(PauliX), &[QubitId::new(0)]).unwrap();
    circuit.insert_operation(1, x_op).unwrap();
    assert_eq!(circuit.len(), 3);
    assert_eq!(circuit.get_operation(1).unwrap().gate().name(), "X");
}

#[test]
fn circuit_clone_independence() {
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    let mut clone = circuit.clone();
    clone
        .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
        .unwrap();
    assert_eq!(circuit.len(), 1);
    assert_eq!(clone.len(), 2);
}

#[test]
fn circuit_clear() {
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
        .unwrap();
    circuit.clear();
    assert!(circuit.is_empty());
}

#[test]
fn circuit_display() {
    let mut circuit = Circuit::new(3);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    let display = format!("{}", circuit);
    assert!(display.contains("3 qubits"));
    assert!(display.contains("1 operations"));
}

// ============================================================================
// 9. GateOp
// ============================================================================

#[test]
fn gate_op_creation_valid() {
    let op = GateOp::new(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)]).unwrap();
    assert_eq!(op.num_qubits(), 2);
    assert_eq!(op.gate().name(), "CNOT");
}

#[test]
fn gate_op_wrong_qubit_count() {
    assert!(GateOp::new(Arc::new(CNot), &[QubitId::new(0)]).is_err());
}

#[test]
fn gate_op_duplicate_qubits() {
    assert!(GateOp::new(Arc::new(CNot), &[QubitId::new(0), QubitId::new(0)]).is_err());
}

#[test]
fn gate_op_display() {
    let op = GateOp::new(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
    let display = format!("{}", op);
    assert!(display.contains("H"));
    assert!(display.contains("q0"));
}

// ============================================================================
// 10. Gate trait properties
// ============================================================================

#[test]
fn gate_metadata() {
    assert_eq!(Hadamard.name(), "H");
    assert_eq!(Hadamard.num_qubits(), 1);
    assert!(Hadamard.is_unitary());
    assert!(Hadamard.is_hermitian());
    assert!(!Hadamard.is_diagonal());
    assert!(Hadamard.matrix().is_some());

    assert!(PauliZ.is_diagonal());
    assert!(PauliZ.is_hermitian());
    assert_eq!(CNot.num_qubits(), 2);
    assert!(CNot.matrix().is_some());
    assert_eq!(Toffoli.num_qubits(), 3);
}

#[test]
fn rotation_gate_properties() {
    let rx = RotationX::new(std::f64::consts::PI);
    assert_eq!(rx.name(), "RX");
    assert_eq!(rx.num_qubits(), 1);
    let _m = rx.matrix();

    let ry = RotationY::new(std::f64::consts::FRAC_PI_2);
    assert_eq!(ry.name(), "RY");

    let rz = RotationZ::new(0.0);
    assert_eq!(rz.name(), "RZ");
    assert!(rz.is_diagonal());

    let phase = Phase::new(std::f64::consts::PI);
    assert_eq!(phase.name(), "P");
    assert!(phase.is_diagonal());
}

// ============================================================================
// 11. QubitId and Qubit<N>
// ============================================================================

#[test]
fn qubit_id_basics() {
    let q = QubitId::new(5);
    assert_eq!(q.index(), 5);
    let q2: QubitId = 3.into();
    assert_eq!(q2.index(), 3);
    let idx: usize = q.into();
    assert_eq!(idx, 5);
}

#[test]
fn qubit_id_equality_ordering() {
    let q0 = QubitId::new(0);
    let q1 = QubitId::new(1);
    assert_eq!(q0, QubitId::new(0));
    assert_ne!(q0, q1);
    assert!(q0 < q1);
}

#[test]
fn qubit_ref_type_safe() {
    let q: Qubit<5> = Qubit::new(3).unwrap();
    assert_eq!(q.index(), 3);
    assert_eq!(Qubit::<5>::circuit_size(), 5);
    assert_eq!(q.to_qubit_id().index(), 3);
}

#[test]
fn qubit_ref_out_of_bounds() {
    let result: Result<Qubit<3>> = Qubit::new(5);
    assert!(result.is_err());
}

// ============================================================================
// 12. Stateful Debugger
// ============================================================================

#[test]
fn stateful_debugger_snapshots() {
    let snapshot = StateSnapshot {
        step: 0,
        num_qubits: 1,
        amplitudes: vec![
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(0.0, 0.0),
        ],
        gate_applied: None,
    };
    assert_eq!(snapshot.num_qubits, 1);
    let probs = snapshot.measurement_probabilities();
    assert!((probs[0].1 - 1.0).abs() < 1e-10);
}

#[test]
fn stateful_debugger_most_likely() {
    let amplitudes = vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
    ];
    let result = StatefulDebugger::most_likely_outcome(&amplitudes, 1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().0, "|1⟩");
}

#[test]
fn stateful_debugger_purity() {
    let amplitudes = vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ];
    let purity = StatefulDebugger::purity(&amplitudes);
    assert!((purity - 1.0).abs() < 1e-10);
}

#[test]
fn stateful_debugger_format_state_vector() {
    let config = VisualizationConfig::default();
    let amplitudes = vec![
        num_complex::Complex64::new(1.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ];
    let formatted = StatefulDebugger::format_state_vector(&amplitudes, 1, &config);
    assert!(!formatted.is_empty());
}

// ============================================================================
// 13. Monte Carlo Samplers
// ============================================================================

#[test]
fn monte_carlo_depolarizing() {
    let channel = DepolarizingChannel::new(0.1).unwrap();
    let mc = DepolarizingMC::new(&channel);
    let idx = mc.sample(0.5);
    let op = mc.get_operation(idx);
    assert!(matches!(
        op,
        PauliOperation::Identity | PauliOperation::X | PauliOperation::Y | PauliOperation::Z
    ));
}

#[test]
fn monte_carlo_depolarizing_no_error() {
    let channel = DepolarizingChannel::new(0.0).unwrap();
    let mc = DepolarizingMC::new(&channel);
    // With p=0, any random value should give Identity
    assert_eq!(mc.sample(0.0), 0);
    assert_eq!(mc.sample(0.5), 0);
    assert_eq!(mc.sample(0.99), 0);
}

#[test]
fn monte_carlo_amplitude_damping() {
    let channel = AmplitudeDamping::new(0.3).unwrap();
    let mc = AmplitudeDampingMC::new(&channel);
    assert_eq!(mc.gamma(), 0.3);
    let idx = mc.sample(0.5);
    let _op = mc.get_operation(idx);
}

#[test]
fn monte_carlo_phase_damping() {
    let channel = PhaseDamping::new(0.2).unwrap();
    let mc = PhaseDampingMC::new(&channel);
    let idx = mc.sample(0.5);
    let _op = mc.get_operation(idx);
}

#[test]
fn monte_carlo_readout_error() {
    let channel = ReadoutError::new(0.05, 0.05).unwrap();
    let mc = ReadoutErrorMC::new(&channel);
    // Measure 0 with random value well above p01 → should stay 0
    let result = mc.apply_to_measurement(false, 0.9);
    assert!(!result);
    // Measure 0 with random value below p01 → should flip to 1
    let result = mc.apply_to_measurement(false, 0.01);
    assert!(result);
}

// ============================================================================
// 14. Time Tracker
// ============================================================================

#[test]
fn time_tracker_basics() {
    let timing = GateTiming::default();
    let mut tracker = QubitTimeTracker::new(3, timing);
    assert_eq!(tracker.num_qubits(), 3);
    assert_eq!(tracker.total_time(), 0.0);
    tracker.apply_single_qubit_gate(0);
    assert_eq!(tracker.qubit_time(0), Some(0.02));
    assert_eq!(tracker.qubit_time(1), Some(0.0));
}

#[test]
fn time_tracker_two_qubit_gate() {
    let timing = GateTiming::default();
    let mut tracker = QubitTimeTracker::new(3, timing);
    tracker.apply_two_qubit_gate(0, 1);
    assert_eq!(tracker.qubit_time(0), Some(0.1));
    assert_eq!(tracker.qubit_time(1), Some(0.1));
    assert_eq!(tracker.qubit_time(2), Some(0.0));
}

#[test]
fn time_tracker_idle_time() {
    let timing = GateTiming::default();
    let mut tracker = QubitTimeTracker::new(3, timing);
    tracker.apply_single_qubit_gate(0); // q0 at 0.02, q1 at 0, q2 at 0
    let idle1 = tracker.idle_time_since_last_operation(1);
    assert!((idle1 - 0.02).abs() < 1e-10); // q1 idle for 0.02
    let idle0 = tracker.idle_time_since_last_operation(0);
    assert!(idle0.abs() < 1e-10); // q0 just operated
}

#[test]
fn time_tracker_sync_all() {
    let timing = GateTiming::default();
    let mut tracker = QubitTimeTracker::new(3, timing);
    tracker.apply_single_qubit_gate(0);
    tracker.synchronize_all_qubits();
    let all_idle = tracker.all_idle_times();
    for idle in &all_idle {
        assert!(idle.abs() < 1e-10);
    }
}

#[test]
fn time_tracker_reset() {
    let timing = GateTiming::default();
    let mut tracker = QubitTimeTracker::new(2, timing);
    tracker.apply_single_qubit_gate(0);
    tracker.reset();
    assert_eq!(tracker.qubit_time(0), Some(0.0));
    assert_eq!(tracker.total_time(), 0.0);
}

#[test]
fn time_tracker_measurement() {
    let timing = GateTiming::default();
    let mut tracker = QubitTimeTracker::new(2, timing);
    tracker.apply_measurement(0);
    assert_eq!(tracker.qubit_time(0), Some(1.0)); // default measurement_time
}

// ============================================================================
// 15. DAG / Dependency Graph
// ============================================================================

#[test]
fn dependency_graph_from_circuit() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[2]).unwrap();
    let circuit = builder.build();

    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    assert!(dag.is_acyclic());
    assert_eq!(dag.depth().unwrap(), 2);
}

#[test]
fn dependency_graph_topological_sort() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliY), &[0]).unwrap();
    let circuit = builder.build();

    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    let order = dag.topological_sort().unwrap();
    assert_eq!(order.len(), 3);
    let pos0 = order.iter().position(|&x| x == 0).unwrap();
    let pos1 = order.iter().position(|&x| x == 1).unwrap();
    let pos2 = order.iter().position(|&x| x == 2).unwrap();
    assert!(pos0 < pos1);
    assert!(pos1 < pos2);
}

#[test]
fn dependency_graph_parallel_layers() {
    let mut builder = DynamicCircuitBuilder::new(4);
    for q in 0..4 {
        builder.apply_gate(Arc::new(Hadamard), &[q]).unwrap();
    }
    let circuit = builder.build();

    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    let layers = dag.compute_parallel_layers().unwrap();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0].len(), 4);
}

#[test]
fn dependency_graph_to_dot() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let circuit = builder.build();

    let dot = circuit.to_dot().unwrap();
    assert!(dot.contains("digraph"));
}

// ============================================================================
// 16. Validation Rules
// ============================================================================

#[test]
fn validation_rules_qubit_usage() {
    use simq_core::validation::rules::QubitUsageRule;
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let circuit = builder.build();
    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    let rule = QubitUsageRule;
    assert!(rule.validate(&circuit, &dag).is_valid);
}

#[test]
fn validation_rules_cycle_detection() {
    use simq_core::validation::rules::CycleDetectionRule;
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();
    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    assert!(CycleDetectionRule.validate(&circuit, &dag).is_valid);
}

#[test]
fn validation_rules_dependency() {
    use simq_core::validation::rules::DependencyValidationRule;
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    let circuit = builder.build();
    let dag = DependencyGraph::from_circuit(&circuit).unwrap();
    assert!(DependencyValidationRule.validate(&circuit, &dag).is_valid);
}

// ============================================================================
// 17. ASCII renderer config
// ============================================================================

#[test]
fn ascii_config_builder() {
    let config = AsciiConfig::builder()
        .max_width(80)
        .style(RenderStyle::Unicode)
        .build();
    assert_eq!(config.max_width, 80);
}

#[test]
fn ascii_render_with_config() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    let circuit = builder.build();
    let config = AsciiConfig {
        max_width: 60,
        ..Default::default()
    };
    let ascii = circuit.to_ascii_with_config(&config);
    assert!(!ascii.is_empty());
}

#[test]
fn ascii_render_detailed() {
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();
    let config = AsciiConfig::default();
    let rendered = render_ascii_detailed(&circuit, &config);
    assert!(!rendered.ascii.is_empty());
    assert!(rendered.qubits > 0);
}

// ============================================================================
// 18. LaTeX renderer config
// ============================================================================

#[test]
fn latex_config_standalone() {
    let config = LatexConfig::default().with_standalone(true);
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();
    let latex = render_latex_with_config(&circuit, &config);
    assert!(latex.contains("documentclass") || latex.contains("document"));
}

#[test]
fn latex_config_no_labels() {
    let config = LatexConfig::default().with_labels(false);
    let mut builder = DynamicCircuitBuilder::new(2);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    let circuit = builder.build();
    let latex = render_latex_with_config(&circuit, &config);
    assert!(!latex.is_empty());
}

// ============================================================================
// 19. Error types coverage
// ============================================================================

#[test]
fn error_display_messages() {
    let e1 = QuantumError::invalid_qubit(5, 3);
    assert!(format!("{}", e1).contains("5"));

    let e2 = QuantumError::invalid_qubit_count("CNOT", 2, 1);
    assert!(format!("{}", e2).contains("CNOT"));

    let e3 = QuantumError::EmptyCircuit;
    assert!(format!("{}", e3).contains("at least one qubit"));

    let e4 = QuantumError::ValidationError("test".to_string());
    assert!(format!("{}", e4).contains("test"));

    let e5 = QuantumError::CycleDetected {
        operations: vec![0, 1],
    };
    assert!(format!("{}", e5).contains("cycle"));

    let e6 = QuantumError::InvalidDependency {
        from: 0,
        to: 1,
        qubit: 0,
    };
    assert!(format!("{}", e6).contains("dependency"));

    let e7 = QuantumError::SerializationError("ser".to_string());
    assert!(format!("{}", e7).contains("ser"));

    let e8 = QuantumError::DeserializationError("deser".to_string());
    assert!(format!("{}", e8).contains("deser"));

    let e9 = QuantumError::UnknownGateType("FooGate".to_string());
    assert!(format!("{}", e9).contains("FooGate"));

    let e10 = QuantumError::VersionMismatch {
        expected: 1,
        actual: 2,
    };
    assert!(format!("{}", e10).contains("1"));

    let e11 = QuantumError::CacheError("cache".to_string());
    assert!(format!("{}", e11).contains("cache"));
}

// ============================================================================
// 20. Bloch Sphere angles
// ============================================================================

#[test]
fn bloch_angles_roundtrip() {
    use num_complex::Complex64;
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let state = [
        Complex64::new(inv_sqrt2, 0.0),
        Complex64::new(inv_sqrt2, 0.0),
    ];
    let bv = BlochVector::from_state(&state);
    let angles = bv.to_angles();
    let bv2 = angles.to_vector();
    assert!((bv.x - bv2.x).abs() < 1e-10);
    assert!((bv.y - bv2.y).abs() < 1e-10);
    assert!((bv.z - bv2.z).abs() < 1e-10);
}

#[test]
fn bloch_sphere_ascii_render() {
    use num_complex::Complex64;
    let state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let bv = BlochVector::from_state(&state);
    let config = BlochRenderConfig::default();
    let ascii = bv.render_ascii_with_config(&config);
    assert!(!ascii.is_empty());
}

// ============================================================================
// 21. Method chaining
// ============================================================================

#[test]
fn builder_method_chaining() {
    let mut builder = CircuitBuilder::<3>::new();
    let [q0, q1, q2] = builder.qubits();
    builder
        .apply_gate(Arc::new(Hadamard), &[q0])
        .unwrap()
        .apply_gate(Arc::new(Hadamard), &[q1])
        .unwrap()
        .apply_gate(Arc::new(Hadamard), &[q2])
        .unwrap()
        .apply_gate(Arc::new(CNot), &[q0, q1])
        .unwrap()
        .apply_gate(Arc::new(CNot), &[q1, q2])
        .unwrap();
    assert_eq!(builder.build().len(), 5);
}

#[test]
fn dynamic_builder_method_chaining() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder
        .apply_gate(Arc::new(Hadamard), &[0])
        .unwrap()
        .apply_gate(Arc::new(Hadamard), &[1])
        .unwrap()
        .apply_gate(Arc::new(CNot), &[0, 1])
        .unwrap();
    assert_eq!(builder.build().len(), 3);
}

// ============================================================================
// 22. Stress & edge cases
// ============================================================================

#[test]
fn stress_1000_gate_circuit() {
    let mut builder = DynamicCircuitBuilder::new(20);
    for i in 0..1000 {
        builder.apply_gate(Arc::new(Hadamard), &[i % 20]).unwrap();
    }
    let circuit = builder.build();
    assert_eq!(circuit.len(), 1000);
    assert!(circuit.validate().is_ok());
}

#[test]
fn stress_deep_single_qubit() {
    let mut builder = DynamicCircuitBuilder::new(1);
    for _ in 0..500 {
        builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
        builder.apply_gate(Arc::new(PauliX), &[0]).unwrap();
    }
    let circuit = builder.build();
    assert_eq!(circuit.len(), 1000);
    assert_eq!(circuit.compute_depth().unwrap(), 1000);
}

#[test]
fn all_qubit_gate_categories() {
    let mut builder = DynamicCircuitBuilder::new(3);
    builder.apply_gate(Arc::new(Hadamard), &[0]).unwrap();
    builder.apply_gate(Arc::new(CNot), &[0, 1]).unwrap();
    builder.apply_gate(Arc::new(Toffoli), &[0, 1, 2]).unwrap();
    let circuit = builder.build();
    assert_eq!(circuit.single_qubit_gate_count(), 1);
    assert_eq!(circuit.two_qubit_gate_count(), 1);
}

#[test]
fn circuit_operations_mut() {
    let mut circuit = Circuit::new(2);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(PauliX), &[QubitId::new(1)])
        .unwrap();
    let ops = circuit.operations_mut();
    assert_eq!(ops.len(), 2);
    ops.clear();
    assert!(circuit.is_empty());
}

#[test]
fn circuit_two_qubit_operations_iter() {
    let mut circuit = Circuit::new(3);
    circuit
        .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
        .unwrap();
    circuit
        .add_gate(Arc::new(CZ), &[QubitId::new(1), QubitId::new(2)])
        .unwrap();
    let two_q: Vec<_> = circuit.two_qubit_operations().collect();
    assert_eq!(two_q.len(), 2);
}

#[test]
fn circuit_single_qubit_operations_iter() {
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
    let single_q: Vec<_> = circuit.single_qubit_operations().collect();
    assert_eq!(single_q.len(), 2);
}

// ============================================================================
// 23. Serialization (feature-gated)
// ============================================================================

#[cfg(feature = "serialization")]
mod serialization_tests {
    use super::*;

    #[test]
    fn json_roundtrip_simple() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let json = circuit.to_json().unwrap();
        let deserialized = Circuit::from_json(&json).unwrap();
        assert_eq!(deserialized.num_qubits(), 2);
        assert_eq!(deserialized.len(), 1);
    }

    #[test]
    fn json_roundtrip_multi_gate() {
        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(2)])
            .unwrap();
        let json = circuit.to_json_pretty().unwrap();
        assert!(json.contains("H"));
        let deserialized = Circuit::from_json(&json).unwrap();
        assert_eq!(deserialized.len(), 3);
    }

    #[test]
    fn binary_roundtrip() {
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        let bytes = circuit.to_bytes().unwrap();
        assert!(!bytes.is_empty());
        let deserialized = Circuit::from_bytes(&bytes).unwrap();
        assert_eq!(deserialized.num_qubits(), 2);
        assert_eq!(deserialized.len(), 1);
    }

    #[test]
    fn cache_key_same_circuit() {
        let mut c1 = Circuit::new(2);
        c1.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        let mut c2 = Circuit::new(2);
        c2.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        assert_eq!(c1.cache_key(), c2.cache_key());
    }

    #[test]
    fn cache_key_different_circuit() {
        let mut c1 = Circuit::new(2);
        c1.add_gate(Arc::new(Hadamard), &[QubitId::new(0)]).unwrap();
        let mut c2 = Circuit::new(2);
        c2.add_gate(Arc::new(PauliX), &[QubitId::new(0)]).unwrap();
        assert_ne!(c1.cache_key(), c2.cache_key());
    }
}
