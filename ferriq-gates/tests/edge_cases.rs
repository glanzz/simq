//! Edge case and error path tests for ferriq-gates

use num_complex::Complex64;
use ferriq_core::gate::Gate;
use ferriq_gates::*;
use std::f64::consts::{PI, SQRT_2};

// ===========================================================================
// CustomGate edge cases
// ===========================================================================

mod custom_gate_edge_cases {
    use super::*;

    #[test]
    fn empty_name_rejected() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let err = CustomGate::new("", 1, matrix, 1e-10).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidName));
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn wrong_element_count() {
        let matrix = vec![Complex64::new(1.0, 0.0); 3];
        let err = CustomGate::new("Bad", 1, matrix, 1e-10).unwrap_err();
        assert!(matches!(
            err,
            CustomGateError::InvalidDimensions {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn nan_in_matrix() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, f64::NAN),
            Complex64::new(1.0, 0.0),
        ];
        let err = CustomGate::new("NaN", 1, matrix, 1e-10).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidValues));
    }

    #[test]
    fn inf_in_matrix() {
        let matrix = vec![
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let err = CustomGate::new("Inf", 1, matrix, 1e-10).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidValues));
    }

    #[test]
    fn non_unitary_rejected() {
        let matrix = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let err = CustomGate::new("NonU", 1, matrix, 1e-10).unwrap_err();
        match err {
            CustomGateError::NotUnitary {
                max_deviation,
                tolerance,
            } => {
                assert!(max_deviation > tolerance);
            },
            _ => panic!("expected NotUnitary, got {:?}", err),
        }
    }

    #[test]
    fn compose_three_gates_chain() {
        let inv = 1.0 / SQRT_2;
        let h = CustomGate::new(
            "H",
            1,
            vec![
                Complex64::new(inv, 0.0),
                Complex64::new(inv, 0.0),
                Complex64::new(inv, 0.0),
                Complex64::new(-inv, 0.0),
            ],
            1e-10,
        )
        .unwrap();

        // H * H * H = H  (since H^2 = I)
        let h2 = h.compose(&h).unwrap();
        let h3 = h2.compose(&h).unwrap();
        let fid = h3.fidelity(h.matrix_vec()).unwrap();
        assert!((fid - 1.0).abs() < 1e-8);
    }

    #[test]
    fn compose_mismatched_qubit_count() {
        let id1 = CustomGate::new("I1", 1, matrix_ops::identity_matrix(2), 1e-10).unwrap();
        let id2 = CustomGate::new("I2", 2, matrix_ops::identity_matrix(4), 1e-10).unwrap();
        let err = id1.compose(&id2).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn controlled_version_preserves_unitarity() {
        let s_gate = CustomGateBuilder::new("S")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
            ])
            .build()
            .unwrap();

        let cs = s_gate.controlled().unwrap();
        assert_eq!(cs.num_qubits(), 2);
        assert_eq!(cs.name(), "CS");
        assert!(cs.is_unitary());
    }

    #[test]
    fn double_controlled_gate() {
        let x = CustomGateBuilder::new("X")
            .matrix_2x2([
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ])
            .build()
            .unwrap();

        let cx = x.controlled().unwrap();
        assert_eq!(cx.num_qubits(), 2);

        let ccx = cx.controlled().unwrap();
        assert_eq!(ccx.num_qubits(), 3);
        assert_eq!(ccx.name(), "CCX");
        assert!(ccx.is_unitary());
    }

    #[test]
    fn adjoint_of_adjoint_is_original() {
        let s = CustomGateBuilder::new("S")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
            ])
            .build()
            .unwrap();

        let s_dag = s.adjoint();
        let s_dag_dag = s_dag.adjoint();
        let fid = s_dag_dag.fidelity(s.matrix_vec()).unwrap();
        assert!((fid - 1.0).abs() < 1e-8);
    }

    #[test]
    fn fidelity_mismatched_size() {
        let gate = CustomGate::new("I", 1, matrix_ops::identity_matrix(2), 1e-10).unwrap();
        let big = matrix_ops::identity_matrix(4);
        let err = gate.fidelity(&big).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn custom_gate_description_default_and_custom() {
        let z = CustomGateBuilder::new("Z")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ])
            .build()
            .unwrap();
        let desc = z.description();
        assert!(desc.contains("Hermitian"));

        let z2 = CustomGateBuilder::new("Z2")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ])
            .description("My custom desc")
            .build()
            .unwrap();
        assert_eq!(z2.description(), "My custom desc");
    }

    #[test]
    fn custom_gate_debug_impl() {
        let gate = CustomGate::new("Test", 1, matrix_ops::identity_matrix(2), 1e-10).unwrap();
        let dbg = format!("{:?}", gate);
        assert!(dbg.contains("Test"));
        assert!(dbg.contains("matrix_size"));
    }

    #[test]
    fn builder_missing_num_qubits() {
        let err = CustomGateBuilder::new("Bad")
            .matrix(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])
            .build()
            .unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn builder_missing_matrix() {
        let err = CustomGateBuilder::new("Bad")
            .num_qubits(1)
            .build()
            .unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn builder_require_hermitian_fails_for_non_hermitian() {
        let err = CustomGateBuilder::new("Phase")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
            ])
            .require_hermitian(true)
            .build()
            .unwrap_err();
        assert!(matches!(err, CustomGateError::NotHermitian { .. }));
    }

    #[test]
    fn builder_4x4_two_qubit_gate() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let gate = CustomGateBuilder::new("CZ")
            .matrix_4x4([
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, Complex64::new(-1.0, 0.0)],
            ])
            .build()
            .unwrap();
        assert_eq!(gate.num_qubits(), 2);
        assert!(gate.is_hermitian());
    }

    #[test]
    fn builder_build_arc() {
        let gate = CustomGateBuilder::new("X")
            .matrix_2x2([
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ])
            .build_arc()
            .unwrap();
        assert_eq!(gate.name(), "X");
    }
}

// ===========================================================================
// CustomGateError Display
// ===========================================================================

mod error_display {
    use super::*;

    #[test]
    fn all_error_variants_display() {
        let errors: Vec<CustomGateError> = vec![
            CustomGateError::NotUnitary {
                max_deviation: 0.5,
                tolerance: 1e-10,
            },
            CustomGateError::InvalidDimensions {
                expected: 2,
                actual: 3,
            },
            CustomGateError::InvalidSize { size: 3 },
            CustomGateError::InvalidValues,
            CustomGateError::InvalidName,
            CustomGateError::InvalidDeterminant {
                determinant_norm: 2.0,
            },
            CustomGateError::NotHermitian { max_deviation: 0.1 },
        ];

        for e in &errors {
            let msg = e.to_string();
            assert!(!msg.is_empty());
        }

        let e: Box<dyn std::error::Error> = Box::new(errors[0].clone());
        assert!(e.to_string().contains("not unitary"));
    }
}

// ===========================================================================
// ParametricCustomGate edge cases
// ===========================================================================

mod parametric_gate_edge_cases {
    use super::*;

    #[test]
    fn parametric_gate_basic_flow() {
        let mut gate = ParametricCustomGateBuilder::new("RX", 1)
            .with_parameters(vec!["theta"])
            .with_matrix_fn(|params| {
                let theta = params[0];
                let c = (theta / 2.0).cos();
                let s = (theta / 2.0).sin();
                vec![
                    Complex64::new(c, 0.0),
                    Complex64::new(0.0, -s),
                    Complex64::new(0.0, -s),
                    Complex64::new(c, 0.0),
                ]
            })
            .build()
            .unwrap();

        gate.set_parameters(vec![PI / 2.0]).unwrap();
        assert_eq!(gate.parameter_names().len(), 1);
        assert_eq!(gate.parameter_names()[0], "theta");
    }

    #[test]
    fn parametric_gate_wrong_param_count() {
        let mut gate = ParametricCustomGateBuilder::new("RX", 1)
            .with_parameters(vec!["theta"])
            .with_matrix_fn(|params| {
                let c = (params[0] / 2.0).cos();
                let s = (params[0] / 2.0).sin();
                vec![
                    Complex64::new(c, 0.0),
                    Complex64::new(0.0, -s),
                    Complex64::new(0.0, -s),
                    Complex64::new(c, 0.0),
                ]
            })
            .build()
            .unwrap();

        let err = gate.set_parameters(vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn parametric_builder_no_params() {
        let err = ParametricCustomGateBuilder::new("Bad", 1)
            .with_matrix_fn(|_| {
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ]
            })
            .build()
            .unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidName));
    }

    #[test]
    fn parametric_builder_no_matrix_fn() {
        let err = ParametricCustomGateBuilder::new("Bad", 1)
            .with_parameters(vec!["x"])
            .build()
            .unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn parametric_to_static() {
        let gate = ParametricCustomGateBuilder::new("RZ", 1)
            .with_parameters(vec!["phi"])
            .with_matrix_fn(|params| {
                let phi = params[0];
                let c = (phi / 2.0).cos();
                let s = (phi / 2.0).sin();
                vec![
                    Complex64::new(c, -s),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(c, s),
                ]
            })
            .build()
            .unwrap();

        let static_gate = gate.to_static_gate().unwrap();
        assert_eq!(static_gate.name(), "RZ");
        assert!(static_gate.is_unitary());
    }

    #[test]
    fn parametric_debug_impl() {
        let gate = ParametricCustomGateBuilder::new("Test", 1)
            .with_parameters(vec!["a"])
            .with_matrix_fn(|_| matrix_ops::identity_matrix(2))
            .build()
            .unwrap();
        let dbg = format!("{:?}", gate);
        assert!(dbg.contains("Test"));
        assert!(dbg.contains("parameters"));
    }
}

// ===========================================================================
// GateRegistry edge cases
// ===========================================================================

mod registry_edge_cases {
    use super::*;

    fn make_z() -> CustomGate {
        CustomGateBuilder::new("Z")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ])
            .build()
            .unwrap()
    }

    #[test]
    fn unregister_nonexistent_returns_none() {
        let mut reg = GateRegistry::new();
        assert!(reg.unregister("no_such_gate").is_none());
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let reg = GateRegistry::new();
        assert!(reg.get("missing").is_none());
    }

    #[test]
    fn duplicate_name_overwrites() {
        let mut reg = GateRegistry::new();
        reg.register("z", make_z());
        reg.register("z", make_z());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn empty_registry_queries() {
        let reg = GateRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.gate_names().is_empty());
        assert!(reg.gates_for_qubits(1).is_empty());
        assert!(reg.list_gates().is_empty());
    }

    #[test]
    fn gates_for_qubits_filters_correctly() {
        let mut reg = GateRegistry::new();
        reg.register("z1", make_z());

        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let cz = CustomGateBuilder::new("CZ2")
            .matrix_4x4([
                [one, zero, zero, zero],
                [zero, one, zero, zero],
                [zero, zero, one, zero],
                [zero, zero, zero, Complex64::new(-1.0, 0.0)],
            ])
            .build()
            .unwrap();
        reg.register("cz", cz);

        assert_eq!(reg.gates_for_qubits(1).len(), 1);
        assert_eq!(reg.gates_for_qubits(2).len(), 1);
        assert_eq!(reg.gates_for_qubits(3).len(), 0);
    }

    #[test]
    fn list_gates_has_correct_info() {
        let mut reg = GateRegistry::new();
        reg.register("z", make_z());
        let info = reg.list_gates();
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].num_qubits, 1);
        assert!(info[0].is_hermitian);
    }

    #[test]
    fn with_capacity_works() {
        let reg = GateRegistry::with_capacity(100);
        assert!(reg.is_empty());
    }

    #[test]
    fn register_arc_and_retrieve() {
        let mut reg = GateRegistry::new();
        let gate = std::sync::Arc::new(make_z());
        reg.register_arc("z_arc", gate.clone());
        assert!(reg.contains("z_arc"));
        let retrieved = reg.get("z_arc").unwrap();
        assert_eq!(retrieved.name(), gate.name());
    }

    #[test]
    fn default_trait() {
        let reg = GateRegistry::default();
        assert!(reg.is_empty());
    }
}

// ===========================================================================
// Matrix operations edge cases
// ===========================================================================

mod matrix_ops_edge_cases {
    use super::*;
    use ferriq_gates::matrix_ops::*;

    #[test]
    fn identity_matrix_sizes() {
        for n in [1, 2, 4, 8] {
            let id = identity_matrix(n);
            assert_eq!(id.len(), n * n);
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (id[i * n + j].re - expected).abs() < 1e-10,
                        "id[{}][{}] = {} expected {}",
                        i,
                        j,
                        id[i * n + j].re,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn matrix_multiply_identity() {
        let id = identity_matrix(4);
        let result = matrix_multiply(&id, &id);
        for i in 0..16 {
            assert!((result[i] - id[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn tensor_product_identity_x_identity() {
        let id = identity_matrix(2);
        let result = tensor_product(&id, &id);
        let id4 = identity_matrix(4);
        for i in 0..16 {
            assert!((result[i] - id4[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn adjoint_of_real_symmetric_is_self() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let adj = matrix_adjoint(&matrix);
        for i in 0..4 {
            assert!((matrix[i] - adj[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn trace_of_identity() {
        let id = identity_matrix(4);
        let tr = matrix_trace(&id);
        assert!((tr.re - 4.0).abs() < 1e-10);
        assert!(tr.im.abs() < 1e-10);
    }

    #[test]
    fn determinant_2x2_identity() {
        let id = ferriq_gates::matrices::IDENTITY;
        let det = matrix_ops::determinant_2x2(&id);
        assert!((det.re - 1.0).abs() < 1e-10);
        assert!(det.im.abs() < 1e-10);
    }

    #[test]
    fn is_unitary_identity_8x8() {
        let id = identity_matrix(8);
        assert!(is_unitary(&id, 1e-10));
    }

    #[test]
    fn non_unitary_detected() {
        let mut matrix = identity_matrix(2);
        matrix[0] = Complex64::new(2.0, 0.0);
        assert!(!is_unitary(&matrix, 1e-10));
    }

    #[test]
    fn is_hermitian_identity_true() {
        let id = identity_matrix(4);
        assert!(is_hermitian(&id, 1e-10));
    }

    #[test]
    fn is_hermitian_non_hermitian() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 0.0),
        ];
        assert!(!is_hermitian(&matrix, 1e-10));
    }

    #[test]
    fn fidelity_identity_with_self() {
        let id = identity_matrix(4);
        let f = matrix_ops::fidelity(&id, &id);
        assert!((f - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fidelity_orthogonal_unitaries() {
        let x = matrix_ops::matrix_to_vec(&ferriq_gates::matrices::PAULI_X);
        let z = matrix_ops::matrix_to_vec(&ferriq_gates::matrices::PAULI_Z);
        let f = matrix_ops::fidelity(&x, &z);
        assert!(f < 1.0);
    }

    #[test]
    fn vec_to_matrix_2x2_roundtrip() {
        let original = ferriq_gates::matrices::HADAMARD;
        let vec = matrix_to_vec(&original);
        let back = vec_to_matrix_2x2(&vec);
        for i in 0..2 {
            for j in 0..2 {
                assert!((original[i][j] - back[i][j]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn embed_single_qubit_in_3_qubit_system() {
        let x = &ferriq_gates::matrices::PAULI_X;
        let embedded = embed_gate_matrix(x, 3, &[1]);
        assert_eq!(embedded.len(), 64); // 8x8
        assert!(is_unitary(&embedded, 1e-10));
    }

    #[test]
    fn circuit_matrix_empty_circuit() {
        let circuit = ferriq_core::Circuit::new(2);
        let matrix = circuit_matrix(&circuit).unwrap();
        let id = identity_matrix(4);
        for i in 0..16 {
            assert!((matrix[i] - id[i]).norm() < 1e-10);
        }
    }
}

// ===========================================================================
// Lookup table edge cases
// ===========================================================================

mod lookup_edge_cases {
    use super::*;
    use ferriq_gates::lookup::*;

    #[test]
    fn default_config() {
        let config = LookupConfig::default();
        let table = RotationLookupTable::new(config);
        let stats = table.stats();
        assert_eq!(stats.num_entries, 1024);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn zero_angle_lookup() {
        let table = RotationLookupTable::default();

        let rx = table.rx_matrix(0.0);
        assert!((rx[0][0].re - 1.0).abs() < 1e-10);
        assert!(rx[0][1].norm() < 1e-10);

        let ry = table.ry_matrix(0.0);
        assert!((ry[0][0].re - 1.0).abs() < 1e-10);

        let rz = table.rz_matrix(0.0);
        assert!((rz[0][0].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn large_angle_fallback() {
        let config = LookupConfig::new().max_angle(PI / 4.0).num_entries(100);
        let table = RotationLookupTable::new(config);

        let rx = table.rx_matrix(PI);
        let direct = ferriq_gates::matrices::rotation_x(PI);
        for i in 0..2 {
            for j in 0..2 {
                assert!((rx[i][j] - direct[i][j]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn negative_angle_symmetry() {
        let table = RotationLookupTable::default();
        let angle = 0.1;

        let rx_pos = table.rx_matrix(angle);
        let rx_neg = table.rx_matrix(-angle);
        assert!((rx_pos[0][0].re - rx_neg[0][0].re).abs() < 1e-6);
    }

    #[test]
    fn interpolation_disabled_vs_enabled() {
        let config_no_interp = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(10)
            .interpolation_enabled(false);
        let config_interp = LookupConfig::new()
            .max_angle(PI / 4.0)
            .num_entries(10)
            .interpolation_enabled(true);

        let table_no = RotationLookupTable::new(config_no_interp);
        let table_yes = RotationLookupTable::new(config_interp);

        let angle = 0.05;
        let rx_no = table_no.rx_matrix(angle);
        let rx_yes = table_yes.rx_matrix(angle);
        // Both should be approximately correct
        let direct = ferriq_gates::matrices::rotation_x(angle);
        let err_no = (rx_no[0][0].re - direct[0][0].re).abs();
        let err_yes = (rx_yes[0][0].re - direct[0][0].re).abs();
        // Interpolation should give a better or equal result
        assert!(err_yes <= err_no + 1e-10);
    }

    #[test]
    fn stats_display() {
        let table = RotationLookupTable::default();
        let stats = table.stats();
        let display = format!("{}", stats);
        assert!(display.contains("Entries"));
        assert!(display.contains("Memory"));
    }
}

// ===========================================================================
// GeneratedAngleCache edge cases
// ===========================================================================

mod generated_cache_edge_cases {
    use super::*;

    #[test]
    fn clifford_t_exact_match() {
        let result = GeneratedAngleCache::rx_clifford_t(PI / 4.0);
        assert!(result.is_some());

        let result = GeneratedAngleCache::ry_clifford_t(PI / 8.0);
        assert!(result.is_some());

        let result = GeneratedAngleCache::rz_clifford_t(PI / 16.0);
        assert!(result.is_some());
    }

    #[test]
    fn clifford_t_miss() {
        let result = GeneratedAngleCache::rx_clifford_t(0.12345);
        assert!(result.is_none());

        let result = GeneratedAngleCache::ry_clifford_t(1.0);
        assert!(result.is_none());

        let result = GeneratedAngleCache::rz_clifford_t(2.0);
        assert!(result.is_none());
    }

    #[test]
    fn qaoa_out_of_range_fallback() {
        let result = GeneratedAngleCache::rx_qaoa(-1.0);
        let direct = ferriq_gates::matrices::rotation_x(-1.0);
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[i][j] - direct[i][j]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn qaoa_rz_out_of_range() {
        let result = GeneratedAngleCache::rz_qaoa(PI + 1.0);
        let direct = ferriq_gates::matrices::rotation_z(PI + 1.0);
        for i in 0..2 {
            for j in 0..2 {
                assert!((result[i][j] - direct[i][j]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn pi_fraction_hit_and_miss() {
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 3.0).is_some());
        assert!(GeneratedAngleCache::rx_pi_fraction(PI / 6.0).is_some());
        assert!(GeneratedAngleCache::rx_pi_fraction(0.123).is_none());
    }

    #[test]
    fn enhanced_cache_all_rotation_axes() {
        let rx = EnhancedUniversalCache::rx(PI / 4.0);
        let ry = EnhancedUniversalCache::ry(PI / 4.0);
        let rz = EnhancedUniversalCache::rz(PI / 4.0);

        let direct_rx = ferriq_gates::matrices::rotation_x(PI / 4.0);
        let direct_ry = ferriq_gates::matrices::rotation_y(PI / 4.0);
        let direct_rz = ferriq_gates::matrices::rotation_z(PI / 4.0);

        for i in 0..2 {
            for j in 0..2 {
                assert!((rx[i][j] - direct_rx[i][j]).norm() < 1e-8);
                assert!((ry[i][j] - direct_ry[i][j]).norm() < 1e-8);
                assert!((rz[i][j] - direct_rz[i][j]).norm() < 1e-8);
            }
        }
    }

    #[test]
    fn enhanced_cache_large_angle_fallback() {
        let angle = 5.0;
        let rx = EnhancedUniversalCache::rx(angle);
        let direct = ferriq_gates::matrices::rotation_x(angle);
        for i in 0..2 {
            for j in 0..2 {
                assert!((rx[i][j] - direct[i][j]).norm() < 1e-10);
            }
        }
    }
}

// ===========================================================================
// Standard gate edge cases
// ===========================================================================

mod standard_gate_edge_cases {
    use super::*;

    #[test]
    fn all_single_qubit_gates_are_unitary() {
        let gates: Vec<Box<dyn Gate>> = vec![
            Box::new(Hadamard),
            Box::new(PauliX),
            Box::new(PauliY),
            Box::new(PauliZ),
            Box::new(SGate),
            Box::new(TGate),
            Box::new(SXGate),
        ];

        for gate in &gates {
            assert_eq!(gate.num_qubits(), 1, "gate {} should be single-qubit", gate.name());
            assert!(gate.is_unitary(), "gate {} should be unitary", gate.name());
            let matrix = gate.matrix().unwrap();
            assert!(
                ferriq_gates::is_unitary(&matrix, 1e-10),
                "gate {} matrix should be unitary",
                gate.name()
            );
        }
    }

    #[test]
    fn multi_qubit_gates_are_unitary() {
        let gates: Vec<Box<dyn Gate>> = vec![Box::new(CNot), Box::new(CZ), Box::new(Swap)];

        for gate in &gates {
            assert_eq!(gate.num_qubits(), 2, "gate {} should be 2-qubit", gate.name());
            let matrix = gate.matrix().unwrap();
            assert!(
                ferriq_gates::is_unitary(&matrix, 1e-10),
                "gate {} matrix should be unitary",
                gate.name()
            );
        }
    }

    #[test]
    fn toffoli_and_fredkin_are_3_qubit() {
        assert_eq!(Toffoli.num_qubits(), 3);
        assert_eq!(Fredkin.num_qubits(), 3);
        assert!(Toffoli.is_unitary());
        assert!(Fredkin.is_unitary());
    }

    #[test]
    fn rotation_gate_at_zero() {
        let rx = RotationX::new(0.0);
        let ry = RotationY::new(0.0);
        let rz = RotationZ::new(0.0);

        for gate in &[&rx as &dyn Gate, &ry, &rz] {
            let matrix = gate.matrix().unwrap();
            let id = matrix_ops::identity_matrix(2);
            for i in 0..4 {
                assert!(
                    (matrix[i] - id[i]).norm() < 1e-10,
                    "rotation at 0 should be identity for {}",
                    gate.name()
                );
            }
        }
    }

    #[test]
    fn rotation_gate_at_2pi_is_minus_identity() {
        let rx = RotationX::new(2.0 * PI);
        let matrix = Gate::matrix(&rx).unwrap();
        // RX(2π) = -I
        assert!((matrix[0].re - (-1.0)).abs() < 1e-10);
        assert!((matrix[3].re - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn hermitian_gates_are_self_adjoint() {
        let hermitian_gates: Vec<Box<dyn Gate>> = vec![
            Box::new(Hadamard),
            Box::new(PauliX),
            Box::new(PauliY),
            Box::new(PauliZ),
        ];

        for gate in &hermitian_gates {
            assert!(gate.is_hermitian(), "{} should be hermitian", gate.name());
            let m = gate.matrix().unwrap();
            assert!(
                ferriq_gates::is_hermitian(&m, 1e-10),
                "{} matrix should be hermitian",
                gate.name()
            );
        }
    }

    #[test]
    fn non_hermitian_gates() {
        assert!(!SGate.is_hermitian());
        assert!(!TGate.is_hermitian());
    }
}

// ===========================================================================
// Validation utilities
// ===========================================================================

mod validation_edge_cases {
    use super::*;
    use ferriq_gates::custom::validation;

    #[test]
    fn validate_quantum_gate_valid() {
        let h_matrix = vec![
            Complex64::new(1.0 / SQRT_2, 0.0),
            Complex64::new(1.0 / SQRT_2, 0.0),
            Complex64::new(1.0 / SQRT_2, 0.0),
            Complex64::new(-1.0 / SQRT_2, 0.0),
        ];
        assert!(validation::validate_quantum_gate(&h_matrix, 1, 1e-10).is_ok());
    }

    #[test]
    fn validate_quantum_gate_invalid() {
        let bad = vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        assert!(validation::validate_quantum_gate(&bad, 1, 1e-10).is_err());
    }

    #[test]
    fn is_trace_preserving_identity() {
        let id = matrix_ops::identity_matrix(2);
        assert!(validation::is_trace_preserving(&id, 1e-10));
    }

    #[test]
    fn check_completeness_empty_set() {
        let err = validation::check_completeness_relation(&[], 1e-10).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }

    #[test]
    fn check_completeness_single_identity() {
        let id = matrix_ops::identity_matrix(2);
        assert!(validation::check_completeness_relation(&[id], 1e-10).is_ok());
    }

    #[test]
    fn check_completeness_mismatched_sizes() {
        let id2 = matrix_ops::identity_matrix(2);
        let id4 = matrix_ops::identity_matrix(4);
        let err = validation::check_completeness_relation(&[id2, id4], 1e-10).unwrap_err();
        assert!(matches!(err, CustomGateError::InvalidDimensions { .. }));
    }
}
