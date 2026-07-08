use approx::assert_relative_eq;
use num_complex::Complex64;
use ferriq_state::CowState;
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

// ============================================================
// Deep branching trees (many levels)
// ============================================================

#[test]
fn test_cow_deep_branching_tree_3_levels() {
    let root = CowState::new(3).unwrap();
    assert_eq!(root.ref_count(), 1);

    let level1_a = root.branch();
    let level1_b = root.branch();
    assert_eq!(root.ref_count(), 3);

    let level2_a1 = level1_a.branch();
    let level2_a2 = level1_a.branch();
    let level2_b1 = level1_b.branch();
    let level2_b2 = level1_b.branch();

    // All share the same data
    assert_eq!(root.ref_count(), 7);
    assert_eq!(level2_a1.ref_count(), 7);
    assert_eq!(level2_b2.ref_count(), 7);

    drop(level2_a1);
    drop(level2_a2);
    assert_eq!(root.ref_count(), 5);

    drop(level2_b1);
    drop(level2_b2);
    assert_eq!(root.ref_count(), 3);
}

#[test]
fn test_cow_deep_branch_mutation_isolates() {
    let root = CowState::new(3).unwrap();
    let level1 = root.branch();
    let level2 = level1.branch();
    let mut level3 = level2.branch();
    assert_eq!(root.ref_count(), 4);

    let stats = level3.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    assert!(stats.copied);

    // level3 has its own copy, rest still share
    assert_eq!(level3.ref_count(), 1);
    assert_eq!(root.ref_count(), 3);

    // Root unchanged
    assert_relative_eq!(root.amplitudes()[0].re, 1.0, epsilon = 1e-10);
    // level3 modified
    assert_relative_eq!(level3.amplitudes()[0].re, FRAC_1_SQRT_2, epsilon = 1e-10);
}

// ============================================================
// Multiple concurrent branches with independent modifications
// ============================================================

#[test]
fn test_cow_concurrent_independent_modifications() {
    let root = CowState::new(2).unwrap();
    let mut branch_h = root.branch();
    let mut branch_x = root.branch();

    // Apply different gates to different branches
    branch_h.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    branch_x.apply_single_qubit_gate(&x_gate(), 0).unwrap();

    // Each should have its own state
    assert_eq!(root.ref_count(), 1);
    assert_eq!(branch_h.ref_count(), 1);
    assert_eq!(branch_x.ref_count(), 1);

    // root is |00⟩
    assert_relative_eq!(root.amplitudes()[0].re, 1.0, epsilon = 1e-10);

    // branch_h is (|00⟩ + |01⟩)/√2
    assert_relative_eq!(branch_h.amplitudes()[0].re, FRAC_1_SQRT_2, epsilon = 1e-10);
    assert_relative_eq!(branch_h.amplitudes()[1].re, FRAC_1_SQRT_2, epsilon = 1e-10);

    // branch_x is |01⟩
    assert_relative_eq!(branch_x.amplitudes()[0].norm(), 0.0, epsilon = 1e-10);
    assert_relative_eq!(branch_x.amplitudes()[1].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_cow_many_branches_all_modified() {
    let root = CowState::new(3).unwrap();
    let mut branches: Vec<CowState> = (0..5).map(|_| root.branch()).collect();
    assert_eq!(root.ref_count(), 6);

    for (i, branch) in branches.iter_mut().enumerate() {
        let qubit = i % 3;
        branch.apply_single_qubit_gate(&hadamard(), qubit).unwrap();
    }

    // All branches should be unique now
    assert_eq!(root.ref_count(), 1);
    for branch in &branches {
        assert_eq!(branch.ref_count(), 1);
    }
}

// ============================================================
// Memory behavior with many clones
// ============================================================

#[test]
fn test_cow_many_clones_share_memory() {
    let root = CowState::new(8).unwrap();
    let clones: Vec<CowState> = (0..100).map(|_| root.clone()).collect();

    assert_eq!(root.ref_count(), 101);
    let stats = root.memory_stats();
    assert_eq!(stats.total_refs, 101);
    assert_eq!(stats.shared_memory, 256 * 16); // 2^8 * sizeof(Complex64)

    drop(clones);
    assert_eq!(root.ref_count(), 1);
}

#[test]
fn test_cow_clone_and_mutate_only_copies_once() {
    let root = CowState::new(4).unwrap();
    let mut branch = root.clone();

    // First mutation copies
    let stats1 = branch.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    assert!(stats1.copied);

    // Subsequent mutations don't copy
    let stats2 = branch.apply_single_qubit_gate(&hadamard(), 1).unwrap();
    assert!(!stats2.copied);

    let stats3 = branch.apply_single_qubit_gate(&hadamard(), 2).unwrap();
    assert!(!stats3.copied);
}

#[test]
fn test_cow_branch_measure_preserves_original() {
    let root = CowState::new(2).unwrap();
    let mut branch = root.branch();

    branch.apply_single_qubit_gate(&hadamard(), 0).unwrap();
    let (outcome, stats) = branch.measure_qubit(0, 0.3).unwrap();
    assert!(!stats.copied); // Already unique from gate application

    assert!(outcome == 0 || outcome == 1);
    // Root unchanged
    assert_relative_eq!(root.amplitudes()[0].re, 1.0, epsilon = 1e-10);
}

#[test]
fn test_cow_fidelity_between_branches() {
    let root = CowState::new(2).unwrap();
    let mut branch = root.branch();

    // Identical states should have fidelity 1
    let fid1 = root.fidelity(&branch).unwrap();
    assert_relative_eq!(fid1, 1.0, epsilon = 1e-10);

    // After modification, fidelity should decrease
    branch.apply_single_qubit_gate(&x_gate(), 0).unwrap();
    let fid2 = root.fidelity(&branch).unwrap();
    assert_relative_eq!(fid2, 0.0, epsilon = 1e-10);
}

#[test]
fn test_cow_reset_on_shared_state_copies() {
    let root = CowState::new(3).unwrap();
    let mut branch = root.branch();

    let stats = branch.reset().unwrap();
    assert!(stats.copied);

    assert_eq!(branch.amplitudes()[0], Complex64::new(1.0, 0.0));
    assert_eq!(root.ref_count(), 1);
    assert_eq!(branch.ref_count(), 1);
}
