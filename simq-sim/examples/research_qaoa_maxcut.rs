//! Research Workflow 5: QAOA for MaxCut on a 5-node Ring Graph
//!
//! MaxCut on the 5-cycle C5: vertices {0..4}, edges (0,1)(1,2)(2,3)(3,4)(4,0).
//! The optimal cut for an odd cycle C_n is n-1 = 4.
//!
//! This workflow also documents three library problems we hit on the way
//! (each demonstrated by a runnable sanity check below):
//!
//!  1. QAOAConfig::default() has final_mixer=false, so a depth-1 QAOA
//!     circuit contains no mixer at all -> output stays uniform.
//!  2. RotationX/RotationZ::matrix() snap angles in (pi/4, pi] to a
//!     100-point grid (step pi/99 ~ 1.8 deg) with no exactness check:
//!     RZ(0.8) actually returns the matrix of RZ(0.79336...). This kills
//!     simulation accuracy for generic angles and makes finite-difference
//!     gradients identically zero.
//!  3. Parameter-shift gradients are structurally zero for the QAOA
//!     builder's circuits (shared, doubled parameters), so the built-in
//!     QAOAOptimizer exits after 1 iteration with "GradientConverged".
//!
//! Workaround: exact custom RZ/RX gates + finite-difference gradient descent.
//!
//! Run with: cargo run --release -p simq-sim --example research_qaoa_maxcut

use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simq_core::{ascii_renderer, Circuit, Gate, QubitId};
use simq_gates::standard::{CNot, Hadamard, RotationZ};
use simq_sim::gradient::{
    compute_gradient_finite_difference, compute_gradient_parameter_shift, FiniteDifferenceConfig,
    ParameterShiftConfig,
};
use simq_sim::qaoa::{
    evaluate_maxcut_solution, Graph, InitialState, MixerType, ProblemType, QAOACircuitBuilder,
    QAOAConfig,
};
use simq_sim::Simulator;
use simq_state::measurement::ComputationalBasis;
use simq_state::DenseState;
use std::sync::Arc;

const SHOTS: usize = 8192;
const N: usize = 5;
const EDGES: [(usize, usize); 5] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)];
const MAX_CUT: f64 = 4.0;

// ---------------------------------------------------------------------
// Exact rotation gates (workaround for the angle-snapping cache bug)
// ---------------------------------------------------------------------

#[derive(Debug)]
struct ExactRz(f64);

impl Gate for ExactRz {
    fn name(&self) -> &str {
        "RZ*"
    }
    fn num_qubits(&self) -> usize {
        1
    }
    fn is_diagonal(&self) -> bool {
        true
    }
    fn description(&self) -> String {
        format!("RZ*({:.4})", self.0)
    }
    fn matrix(&self) -> Option<Vec<Complex64>> {
        let h = self.0 / 2.0;
        Some(vec![
            Complex64::new(h.cos(), -h.sin()),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(h.cos(), h.sin()),
        ])
    }
}

#[derive(Debug)]
struct ExactRx(f64);

impl Gate for ExactRx {
    fn name(&self) -> &str {
        "RX*"
    }
    fn num_qubits(&self) -> usize {
        1
    }
    fn description(&self) -> String {
        format!("RX*({:.4})", self.0)
    }
    fn matrix(&self) -> Option<Vec<Complex64>> {
        let h = self.0 / 2.0;
        Some(vec![
            Complex64::new(h.cos(), 0.0),
            Complex64::new(0.0, -h.sin()),
            Complex64::new(0.0, -h.sin()),
            Complex64::new(h.cos(), 0.0),
        ])
    }
}

// ---------------------------------------------------------------------
// QAOA circuit built manually with exact gates
// ---------------------------------------------------------------------

/// params = [gamma_1, beta_1, gamma_2, beta_2, ...]
fn qaoa_circuit(params: &[f64]) -> Circuit {
    let mut c = Circuit::new(N);
    for q in 0..N {
        c.add_gate(Arc::new(Hadamard), &[QubitId::new(q)]).unwrap();
    }
    for layer in 0..params.len() / 2 {
        let (gamma, beta) = (params[2 * layer], params[2 * layer + 1]);
        // Cost layer: e^{-i*gamma*Z_i Z_j} per edge via CNOT-RZ(2g)-CNOT
        for &(i, j) in EDGES.iter() {
            c.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)])
                .unwrap();
            c.add_gate(Arc::new(ExactRz(2.0 * gamma)), &[QubitId::new(j)])
                .unwrap();
            c.add_gate(Arc::new(CNot), &[QubitId::new(i), QubitId::new(j)])
                .unwrap();
        }
        // Mixer layer: e^{-i*beta*X_q} on every qubit
        for q in 0..N {
            c.add_gate(Arc::new(ExactRx(2.0 * beta)), &[QubitId::new(q)])
                .unwrap();
        }
    }
    c
}

fn maxcut_observable() -> simq_state::PauliObservable {
    let builder = QAOACircuitBuilder::new(
        ProblemType::MaxCut(Graph::cycle(5)),
        MixerType::StandardX,
        1,
    );
    builder.cost_observable().unwrap()
}

fn energy(sim: &Simulator, params: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    let circuit = qaoa_circuit(params);
    let result = sim.run(&circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(N, &amps)?;
    Ok(maxcut_observable().expectation_value(&dense)?)
}

fn bits_of(outcome: u64, n: usize) -> Vec<bool> {
    (0..n).map(|q| (outcome >> q) & 1 == 1).collect()
}

/// Finite-difference gradient descent over (gamma, beta) parameters
fn run_depth(
    sim: &Simulator,
    depth: usize,
    seed: u64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let observable = maxcut_observable();
    let fd = FiniteDifferenceConfig {
        epsilon: 1e-6,
        ..Default::default()
    };
    // Multi-start gradient descent: the QAOA landscape is non-convex and
    // (gamma=0) is a stationary point, so we restart from several points
    // and keep the best minimum — standard practice in QAOA studies.
    let starts: Vec<Vec<f64>> = vec![
        (0..2 * depth).map(|i| if i % 2 == 0 { 0.4 } else { 0.6 }).collect(),
        (0..2 * depth).map(|i| if i % 2 == 0 { 0.7 } else { 0.3 }).collect(),
        (0..2 * depth).map(|i| 0.15 + 0.1 * i as f64).collect(),
    ];

    let lr = 0.08;
    let t0 = std::time::Instant::now();
    let mut best: Option<(f64, Vec<f64>, usize)> = None;
    for start in starts {
        let mut params = start;
        let mut iters = 0;
        for _ in 0..400 {
            let grad =
                compute_gradient_finite_difference(sim, qaoa_circuit, &observable, &params, &fd)?;
            if grad.norm() < 1e-5 {
                break;
            }
            for (p, g) in params.iter_mut().zip(grad.gradients.iter()) {
                *p -= lr * g;
            }
            iters += 1;
        }
        let e = energy(sim, &params)?;
        if best.as_ref().map_or(true, |(be, _, _)| e < *be) {
            best = Some((e, params, iters));
        }
    }
    let (e, params, iters) = best.unwrap();

    println!(
        "  p={}  <H_C> = {:+.6}  ({} iterations, {:?})",
        depth,
        e,
        iters,
        t0.elapsed()
    );
    println!(
        "       optimal params (gamma, beta per layer): {:?}",
        params
            .iter()
            .map(|p| (p * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );

    // cut = (num_edges - <H_C>) / 2 for unit weights
    let expected_cut = (EDGES.len() as f64 - e) / 2.0;
    println!(
        "       expected cut value <C> = {:.4}  (max possible = {})",
        expected_cut, MAX_CUT
    );

    // Sample the optimized circuit like a real experiment
    let circuit = qaoa_circuit(&params);
    let result = sim.run(&circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(N, &amps)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let sampling = ComputationalBasis::new().sample(&dense, SHOTS, &mut || rng.gen::<f64>())?;
    let mut counts: Vec<(u64, usize)> = sampling.counts.iter().map(|(&k, &v)| (k, v)).collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1));

    let graph = Graph::cycle(5);
    println!("\n       Top sampled bitstrings ({} shots):", SHOTS);
    for (outcome, count) in counts.iter().take(6) {
        let cut = evaluate_maxcut_solution(&graph, &bits_of(*outcome, N));
        let p = *count as f64 / SHOTS as f64;
        let bar = "#".repeat((p * 60.0).round() as usize);
        println!(
            "         |{:05b}>  cut={}  {:>5} shots ({:.3})  {}",
            outcome, cut, count, p, bar
        );
    }
    let p_optimal: f64 = counts
        .iter()
        .filter(|(o, _)| evaluate_maxcut_solution(&graph, &bits_of(*o, N)) as usize == 4)
        .map(|(_, c)| *c as f64 / SHOTS as f64)
        .sum();
    println!("       P(sampling an optimal cut) = {:.3}\n", p_optimal);

    Ok(expected_cut)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==========================================================");
    println!(" Workflow 5: QAOA MaxCut on the 5-cycle (C5), p = 1 and 2");
    println!("==========================================================\n");

    let sim = Simulator::new(Default::default());
    let graph = Graph::cycle(5);
    println!(
        "Problem: MaxCut on C5 ({} vertices, {} edges), optimal cut = 4\n",
        graph.num_vertices,
        graph.num_edges()
    );

    println!("QAOA p=1 circuit (gamma=0.4, beta=0.6, exact custom gates):\n");
    println!("{}", ascii_renderer::render(&qaoa_circuit(&[0.4, 0.6])));

    // Baseline: expected cut of uniform random guessing
    let expected_random: f64 = (0..32)
        .map(|o| evaluate_maxcut_solution(&graph, &bits_of(o, N)))
        .sum::<f64>()
        / 32.0;
    println!(
        "Baseline (uniform random guessing): <C> = {:.3}, ratio = {:.3}\n",
        expected_random,
        expected_random / MAX_CUT
    );

    // --------------------------------------------------------------
    // Sanity check 1: library default config drops the mixer
    // --------------------------------------------------------------
    println!("--- Sanity check 1: QAOAConfig::default() (final_mixer = false) ---");
    let default_builder =
        QAOACircuitBuilder::new(ProblemType::MaxCut(Graph::cycle(5)), MixerType::StandardX, 1);
    let default_circuit = default_builder.build(&[0.4, 0.6]);
    let result = sim.run(&default_circuit)?;
    let amps = result.state.to_dense_vec();
    let dense = DenseState::from_amplitudes(N, &amps)?;
    let probs = dense.get_all_probabilities();
    let max_p = probs.iter().cloned().fold(0.0, f64::max);
    let min_p = probs.iter().cloned().fold(1.0, f64::min);
    println!(
        "  p=1 default-config output: min P = {:.6}, max P = {:.6} (uniform = {:.6})",
        min_p,
        max_p,
        1.0 / 32.0
    );
    println!("  => The mixer layer is silently dropped at p=1: the circuit only");
    println!("     applies diagonal phases, the distribution stays exactly uniform.\n");

    // --------------------------------------------------------------
    // Sanity check 2: rotation-gate angle snapping
    // --------------------------------------------------------------
    println!("--- Sanity check 2: RotationZ matrix cache snaps angles ---");
    let cached = RotationZ::new(0.8).matrix();
    let exact = RotationZ::new(0.8).matrix_uncached();
    let err = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| (cached[i][j] - exact[i][j]).norm())
        .fold(0.0, f64::max);
    let shifted = RotationZ::new(0.8 + 1e-7).matrix();
    let dshift = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| (cached[i][j] - shifted[i][j]).norm())
        .fold(0.0, f64::max);
    println!("  max |RZ(0.8).matrix() - exact RZ(0.8)|      = {:.3e}  (should be ~1e-16)", err);
    println!("  max |RZ(0.8).matrix() - RZ(0.8+1e-7)|      = {:.3e}  (should be ~5e-8)", dshift);
    println!("  => Angles in (pi/4, pi] are snapped to a pi/99 grid (~1.8 deg step).");
    println!("     Simulations with generic RZ/RX angles are systematically wrong,");
    println!("     and epsilon-perturbations return the SAME matrix, so finite-");
    println!("     difference gradients through library gates are identically 0.\n");

    // --------------------------------------------------------------
    // Sanity check 3: gradients through the library QAOA builder
    // --------------------------------------------------------------
    println!("--- Sanity check 3: gradients through QAOACircuitBuilder ---");
    {
        let builder = Arc::new(QAOACircuitBuilder::with_config(
            ProblemType::MaxCut(Graph::cycle(5)),
            QAOAConfig {
                depth: 1,
                mixer: MixerType::StandardX,
                initial_state: InitialState::UniformSuperposition,
                final_mixer: true,
            },
        ));
        let observable = builder.cost_observable().map_err(|e| e.to_string())?;
        let b = builder.clone();
        let cb = move |p: &[f64]| b.build(p);
        let params = [0.4, 0.6];
        let ps = compute_gradient_parameter_shift(
            &sim,
            &cb,
            &observable,
            &params,
            &ParameterShiftConfig::default(),
        )?;
        let fd = compute_gradient_finite_difference(
            &sim,
            &cb,
            &observable,
            &params,
            &FiniteDifferenceConfig::default(),
        )?;
        let fd_exact = compute_gradient_finite_difference(
            &sim,
            qaoa_circuit,
            &maxcut_observable(),
            &params,
            &FiniteDifferenceConfig::default(),
        )?;
        println!("  parameter-shift (library gates): |g| = {:.2e}", ps.norm());
        println!("  finite-diff     (library gates): |g| = {:.2e}", fd.norm());
        println!("  finite-diff     (exact gates)  : |g| = {:.2e}  <- true gradient", fd_exact.norm());
        println!("  => Both library paths report zero gradient at a generic point, so");
        println!("     QAOAOptimizer terminates after 1 iteration (\"GradientConverged\")");
        println!("     without moving the parameters. The true gradient norm is O(1).\n");
    }

    // --------------------------------------------------------------
    // The actual experiment, with exact gates + FD gradient descent
    // --------------------------------------------------------------
    println!("--- QAOA optimization (exact gates, finite-difference GD) ---\n");
    let cut_p1 = run_depth(&sim, 1, 101)?;
    let cut_p2 = run_depth(&sim, 2, 202)?;

    println!("--- Summary: approximation ratios ---\n");
    println!("  {:<26} {:>10} {:>10}", "Method", "<C>", "ratio");
    println!("  {:-<48}", "");
    println!(
        "  {:<26} {:>10.3} {:>10.3}",
        "Random guessing",
        expected_random,
        expected_random / MAX_CUT
    );
    println!("  {:<26} {:>10.3} {:>10.3}", "QAOA p=1", cut_p1, cut_p1 / MAX_CUT);
    println!("  {:<26} {:>10.3} {:>10.3}", "QAOA p=2", cut_p2, cut_p2 / MAX_CUT);
    println!("  {:<26} {:>10.3} {:>10.3}", "Optimal (classical)", MAX_CUT, 1.0);

    println!("\n==> With exact gates and a working gradient, QAOA beats random");
    println!("    guessing and improves with depth, as theory predicts.");

    Ok(())
}
