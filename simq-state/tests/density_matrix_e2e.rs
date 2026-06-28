use approx::assert_relative_eq;
use num_complex::Complex64;
use simq_state::DensityMatrix;
use std::f64::consts::FRAC_1_SQRT_2;

// ============================================================
// Multiple consecutive Kraus channels
// ============================================================

fn bit_flip_kraus(p: f64) -> Vec<(Vec<Complex64>, usize)> {
    let sqrt_p = p.sqrt();
    let sqrt_1_p = (1.0 - p).sqrt();
    vec![
        (
            vec![
                Complex64::new(sqrt_1_p, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_1_p, 0.0),
            ],
            2,
        ),
        (
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_p, 0.0),
                Complex64::new(sqrt_p, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            2,
        ),
    ]
}

fn phase_damping_kraus(gamma: f64) -> Vec<(Vec<Complex64>, usize)> {
    let sqrt_gamma = gamma.sqrt();
    let sqrt_1_gamma = (1.0 - gamma).sqrt();
    vec![
        (
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_1_gamma, 0.0),
            ],
            2,
        ),
        (
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_gamma, 0.0),
            ],
            2,
        ),
    ]
}

#[test]
fn test_dm_consecutive_bit_flip_channels() {
    // Use |0⟩, not |+⟩ (|+⟩ is an eigenstate of X so bit-flip doesn't reduce purity)
    let mut dm = DensityMatrix::new(1).unwrap();

    let bf = bit_flip_kraus(0.1);
    dm.apply_kraus_channel(&bf, &[0]).unwrap();
    assert!(dm.is_valid(1e-8));
    assert!((dm.trace() - 1.0).abs() < 1e-8);

    dm.apply_kraus_channel(&bf, &[0]).unwrap();
    assert!(dm.is_valid(1e-8));
    assert!((dm.trace() - 1.0).abs() < 1e-8);

    dm.apply_kraus_channel(&bf, &[0]).unwrap();
    assert!(dm.is_valid(1e-8));
    assert!(dm.purity() < 1.0);
}

#[test]
fn test_dm_bit_flip_then_phase_damping() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let mut dm = DensityMatrix::from_state_vector(1, &amps).unwrap();

    let bf = bit_flip_kraus(0.1);
    dm.apply_kraus_channel(&bf, &[0]).unwrap();

    let pd = phase_damping_kraus(0.2);
    dm.apply_kraus_channel(&pd, &[0]).unwrap();

    assert!(dm.is_valid(1e-8));
    assert!(dm.purity() < 1.0);
}

// ============================================================
// Error accumulation over many operations
// ============================================================

#[test]
fn test_dm_many_unitaries_preserve_purity() {
    let h = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];

    let mut dm = DensityMatrix::new(1).unwrap();
    for _ in 0..20 {
        dm.apply_unitary(&h, &[0]).unwrap();
    }

    assert_relative_eq!(dm.purity(), 1.0, epsilon = 1e-8);
    assert_relative_eq!(dm.trace(), 1.0, epsilon = 1e-8);
}

#[test]
fn test_dm_repeated_noise_reduces_purity() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let mut dm = DensityMatrix::from_state_vector(1, &amps).unwrap();

    let pd = phase_damping_kraus(0.05);
    let mut purity_prev = dm.purity();
    for _ in 0..10 {
        dm.apply_kraus_channel(&pd, &[0]).unwrap();
        let purity_now = dm.purity();
        assert!(purity_now <= purity_prev + 1e-10);
        purity_prev = purity_now;
    }

    assert!(dm.purity() < 0.9);
    assert!(dm.is_valid(1e-8));
}

// ============================================================
// Reduced density matrix properties (positivity, trace preservation)
// ============================================================

#[test]
fn test_dm_partial_trace_preserves_trace() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let dm = DensityMatrix::from_state_vector(2, &amps).unwrap();

    let reduced = dm.partial_trace(&[0]).unwrap();
    assert_relative_eq!(reduced.trace(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_dm_partial_trace_bell_state_maximally_mixed() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let dm = DensityMatrix::from_state_vector(2, &amps).unwrap();

    let reduced = dm.partial_trace(&[0]).unwrap();
    assert_relative_eq!(reduced.purity(), 0.5, epsilon = 1e-10);
    assert_relative_eq!(reduced.get(0, 0).re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(reduced.get(1, 1).re, 0.5, epsilon = 1e-10);
}

#[test]
fn test_dm_partial_trace_product_state_pure() {
    let amps = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let dm = DensityMatrix::from_state_vector(2, &amps).unwrap();

    let reduced = dm.partial_trace(&[0]).unwrap();
    assert_relative_eq!(reduced.purity(), 1.0, epsilon = 1e-10);
    assert_relative_eq!(reduced.trace(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_dm_partial_trace_hermiticity() {
    let amps = vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.5),
        Complex64::new(0.0, 0.5),
        Complex64::new(0.0, 0.0),
    ];
    let dm = DensityMatrix::from_state_vector(2, &amps).unwrap();

    let reduced = dm.partial_trace(&[1]).unwrap();
    assert!(reduced.is_valid(1e-8));
}

// ============================================================
// Measurement on density matrix
// ============================================================

#[test]
fn test_dm_measurement_preserves_validity() {
    let amps = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];
    let mut dm = DensityMatrix::from_state_vector(1, &amps).unwrap();

    let outcome = dm.measure(0, 0.3).unwrap();
    let _ = outcome;
    assert!(dm.is_valid(1e-8));
    assert_relative_eq!(dm.purity(), 1.0, epsilon = 1e-8);
}

#[test]
fn test_dm_measurement_deterministic_for_basis_state() {
    let mut dm = DensityMatrix::new(1).unwrap();
    let outcome = dm.measure(0, 0.5).unwrap();
    assert!(!outcome); // |0⟩ always measures 0
}

// ============================================================
// Maximally mixed state properties
// ============================================================

#[test]
fn test_dm_maximally_mixed_properties() {
    let dm = DensityMatrix::maximally_mixed(2).unwrap();

    assert_relative_eq!(dm.trace(), 1.0, epsilon = 1e-10);
    assert_relative_eq!(dm.purity(), 0.25, epsilon = 1e-10);
    assert!(dm.is_valid(1e-10));

    for i in 0..4 {
        assert_relative_eq!(dm.get(i, i).re, 0.25, epsilon = 1e-10);
        for j in 0..4 {
            if i != j {
                assert_relative_eq!(dm.get(i, j).norm(), 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_dm_maximally_mixed_unitary_invariant() {
    let mut dm = DensityMatrix::maximally_mixed(1).unwrap();
    let purity_before = dm.purity();

    let h = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(-FRAC_1_SQRT_2, 0.0),
    ];
    dm.apply_unitary(&h, &[0]).unwrap();

    assert_relative_eq!(dm.purity(), purity_before, epsilon = 1e-10);
    assert_relative_eq!(dm.get(0, 0).re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(dm.get(1, 1).re, 0.5, epsilon = 1e-10);
}
