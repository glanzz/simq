use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::{ComputationalBasis, DenseState, MidCircuitMeasurement};
use std::f64::consts::FRAC_1_SQRT_2;

fn hadamard() -> [[Complex64; 2]; 2] {
    let h = FRAC_1_SQRT_2;
    [
        [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
        [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
    ]
}

fn x_gate() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ]
}

struct TestRng {
    state: u64,
}

impl TestRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as f64 / 32768.0
    }
}

// ============================================================
// Conditional measurements (deterministic outcomes)
// ============================================================

#[test]
fn test_measurement_deterministic_zero_state() {
    let mut state = DenseState::new(3).unwrap();
    let measurement = ComputationalBasis::new();
    let mut rng = TestRng::new(42);

    let result = measurement
        .measure_once(&mut state, &mut || rng.next())
        .unwrap();
    assert_eq!(result.outcome, 0);
    assert_relative_eq!(result.probability, 1.0);
}

#[test]
fn test_measurement_deterministic_one_state() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    state.apply_single_qubit_gate(&x_gate(), 1).unwrap();

    let measurement = ComputationalBasis::new();
    let mut rng = TestRng::new(42);

    let result = measurement
        .measure_once(&mut state, &mut || rng.next())
        .unwrap();
    assert_eq!(result.outcome, 3); // |11⟩
    assert_relative_eq!(result.probability, 1.0);
}

#[test]
fn test_measurement_conditional_on_rng_value() {
    let amplitudes = vec![
        Complex64::new(0.6, 0.0),
        Complex64::new(0.8, 0.0),
    ];
    let mut state0 = DenseState::from_amplitudes(1, &amplitudes).unwrap();
    let mut state1 = DenseState::from_amplitudes(1, &amplitudes).unwrap();

    let measurement = ComputationalBasis::new();

    // rng < 0.36 -> outcome 0
    let result0 = measurement
        .measure_once(&mut state0, &mut || 0.1)
        .unwrap();
    assert_eq!(result0.outcome, 0);

    // rng > 0.36 -> outcome 1
    let result1 = measurement
        .measure_once(&mut state1, &mut || 0.5)
        .unwrap();
    assert_eq!(result1.outcome, 1);
}

// ============================================================
// Measurement-induced entanglement breaking verification
// ============================================================

#[test]
fn test_measurement_breaks_bell_entanglement() {
    // Bell state (|00⟩ + |11⟩)/√2
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let mut state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    let mid = MidCircuitMeasurement::new(vec![0]);
    let outcomes = mid.measure(&mut state, &mut || 0.3).unwrap();
    let measured = outcomes[0].1;

    // After measuring qubit 0, the state should be a product state
    let rho = state.partial_trace(&[0]).unwrap();
    // Should be pure (purity = 1) since entanglement is broken
    let purity = rho[0].re * rho[0].re + rho[1].norm_sqr() + rho[2].norm_sqr() + rho[3].re * rho[3].re;
    // Actually trace(rho^2) for 2x2
    let purity_correct = (rho[0] * rho[0] + rho[1] * rho[2]).re + (rho[2] * rho[1] + rho[3] * rho[3]).re;
    assert_relative_eq!(purity_correct, 1.0, epsilon = 1e-8);

    if measured == 0 {
        assert_relative_eq!(state.amplitudes()[0].norm(), 1.0, epsilon = 1e-10);
    } else {
        assert_relative_eq!(state.amplitudes()[3].norm(), 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_measurement_ghz_breaks_into_basis_state() {
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); 8];
    amplitudes[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
    amplitudes[7] = Complex64::new(FRAC_1_SQRT_2, 0.0);
    let mut state = DenseState::from_amplitudes(3, &amplitudes).unwrap();

    let mid = MidCircuitMeasurement::new(vec![0, 1, 2]);
    let outcomes = mid.measure(&mut state, &mut || 0.3).unwrap();

    // All qubits should have same value
    let val = outcomes[0].1;
    for (_, v) in &outcomes {
        assert_eq!(*v, val);
    }

    // State should be a computational basis state
    assert!(state.is_normalized(1e-10));
    let non_zero_count = state
        .amplitudes()
        .iter()
        .filter(|a| a.norm_sqr() > 1e-10)
        .count();
    assert_eq!(non_zero_count, 1);
}

// ============================================================
// Post-selection statistics
// ============================================================

#[test]
fn test_sampling_statistics_match_probabilities() {
    let amplitudes = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = TestRng::new(12345);

    let result = measurement
        .sample(&state, 10000, &mut || rng.next())
        .unwrap();

    for outcome in 0..4u64 {
        let freq = result.get_probability(outcome);
        assert!((freq - 0.25).abs() < 0.03, "Outcome {} freq: {}", outcome, freq);
    }
}

#[test]
fn test_sampling_biased_state() {
    let amplitudes = vec![
        Complex64::new((0.8_f64).sqrt(), 0.0),
        Complex64::new((0.2_f64).sqrt(), 0.0),
    ];
    let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = TestRng::new(42);

    let result = measurement
        .sample(&state, 10000, &mut || rng.next())
        .unwrap();

    let p0 = result.get_probability(0);
    let p1 = result.get_probability(1);
    assert!((p0 - 0.8).abs() < 0.03, "p0 = {}", p0);
    assert!((p1 - 0.2).abs() < 0.03, "p1 = {}", p1);
}

#[test]
fn test_sampling_result_bitstring_counts() {
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    let measurement = ComputationalBasis::new().with_collapse(false);
    let mut rng = TestRng::new(99);

    let result = measurement
        .sample(&state, 1000, &mut || rng.next())
        .unwrap();

    let bitstrings = result.to_bitstring_counts(2);
    let count_00 = bitstrings.get("00").copied().unwrap_or(0);
    let count_11 = bitstrings.get("11").copied().unwrap_or(0);
    assert!(count_00 > 300 && count_00 < 700);
    assert!(count_11 > 300 && count_11 < 700);

    let count_01 = bitstrings.get("01").copied().unwrap_or(0);
    let count_10 = bitstrings.get("10").copied().unwrap_or(0);
    assert!(count_01 < 30);
    assert!(count_10 < 30);
}

// ============================================================
// Mid-circuit measurement advanced
// ============================================================

#[test]
fn test_mid_circuit_preserves_unmeasured_superposition() {
    // |+⟩ ⊗ |0⟩: qubit 0 in superposition, qubit 1 in |0⟩
    let amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let mut state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

    let mid = MidCircuitMeasurement::new(vec![1]);
    let outcomes = mid.measure(&mut state, &mut || 0.3).unwrap();

    // Qubit 1 should measure 0
    assert_eq!(outcomes[0].1, 0);

    // Qubit 0 should still be in superposition
    assert_relative_eq!(state.amplitudes()[0].norm(), FRAC_1_SQRT_2, epsilon = 1e-10);
    assert_relative_eq!(state.amplitudes()[1].norm(), FRAC_1_SQRT_2, epsilon = 1e-10);
}

#[test]
fn test_mid_circuit_measure_with_state_returns_clone() {
    let mut state = DenseState::new(2).unwrap();
    state.apply_single_qubit_gate(&hadamard(), 0).unwrap();

    let mid = MidCircuitMeasurement::new(vec![0]);
    let (outcomes, final_state) = mid
        .measure_with_state(&mut state, &mut || 0.3)
        .unwrap();

    assert_eq!(outcomes.len(), 1);
    assert!(final_state.is_normalized(1e-10));
}
