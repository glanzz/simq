use ferriq_sim::qaoa::{
    evaluate_maxcut_solution, evaluate_partition_solution, random_initial_parameters, Graph,
    InitialState, MixerType, ProblemType, QAOACircuitBuilder, QAOAConfig,
};
use ferriq_sim::{Simulator, SimulatorConfig};
use ferriq_state::observable::PauliObservable;
use ferriq_state::AdaptiveState;

fn make_sim() -> Simulator {
    Simulator::new(SimulatorConfig::default().with_optimization(false))
}

fn expectation(sim: &Simulator, circuit: &ferriq_core::Circuit, obs: &PauliObservable) -> f64 {
    let result = sim.run(circuit).unwrap();
    match &result.state {
        AdaptiveState::Dense(d) => obs.expectation_value(d).unwrap(),
        AdaptiveState::Sparse { state: s, .. } => {
            let d = ferriq_state::DenseState::from_sparse(s).unwrap();
            obs.expectation_value(&d).unwrap()
        },
    }
}

// ============================================================================
// Graph construction tests
// ============================================================================

#[test]
fn graph_from_edges_basic() {
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 2.0)]);
    assert_eq!(g.num_vertices, 3);
    assert_eq!(g.num_edges(), 2);
    assert_eq!(g.degree(0), 1);
    assert_eq!(g.degree(1), 2);
    assert_eq!(g.degree(2), 1);
}

#[test]
fn graph_from_edges_empty() {
    let g = Graph::from_edges(5, &[]);
    assert_eq!(g.num_vertices, 5);
    assert_eq!(g.num_edges(), 0);
    for i in 0..5 {
        assert_eq!(g.degree(i), 0);
        assert!(g.neighbors(i).is_empty());
    }
}

#[test]
fn graph_complete() {
    let g = Graph::complete(5);
    assert_eq!(g.num_vertices, 5);
    assert_eq!(g.num_edges(), 10);
    for i in 0..5 {
        assert_eq!(g.degree(i), 4);
    }
}

#[test]
fn graph_cycle() {
    let g = Graph::cycle(6);
    assert_eq!(g.num_vertices, 6);
    assert_eq!(g.num_edges(), 6);
    for i in 0..6 {
        assert_eq!(g.degree(i), 2);
    }
}

#[test]
fn graph_path() {
    let g = Graph::path(5);
    assert_eq!(g.num_vertices, 5);
    assert_eq!(g.num_edges(), 4);
    assert_eq!(g.degree(0), 1);
    assert_eq!(g.degree(1), 2);
    assert_eq!(g.degree(4), 1);
}

#[test]
fn graph_star() {
    let g = Graph::star(5);
    assert_eq!(g.num_vertices, 5);
    assert_eq!(g.num_edges(), 4);
    assert_eq!(g.degree(0), 4);
    for i in 1..5 {
        assert_eq!(g.degree(i), 1);
    }
}

#[test]
fn graph_grid() {
    let g = Graph::grid(3, 3);
    assert_eq!(g.num_vertices, 9);
    assert_eq!(g.num_edges(), 12);
    assert_eq!(g.degree(0), 2);
    assert_eq!(g.degree(1), 3);
    assert_eq!(g.degree(4), 4);
}

#[test]
fn graph_neighbors_weighted() {
    let g = Graph::from_edges(3, &[(0, 1, 1.5), (0, 2, 2.5)]);
    let neighbors = g.neighbors(0);
    assert_eq!(neighbors.len(), 2);
    let weights: Vec<f64> = neighbors.iter().map(|&(_, w)| w).collect();
    assert!(weights.contains(&1.5));
    assert!(weights.contains(&2.5));
}

// ============================================================================
// ProblemType tests
// ============================================================================

#[test]
fn problem_type_num_qubits_maxcut() {
    let g = Graph::from_edges(5, &[(0, 1, 1.0)]);
    let p = ProblemType::MaxCut(g);
    assert_eq!(p.num_qubits(), 5);
}

#[test]
fn problem_type_num_qubits_partition() {
    let p = ProblemType::NumberPartitioning(vec![1.0, 2.0, 3.0]);
    assert_eq!(p.num_qubits(), 3);
}

#[test]
fn problem_type_num_qubits_coloring() {
    let g = Graph::from_edges(4, &[(0, 1, 1.0)]);
    let p = ProblemType::GraphColoring(g, 3);
    assert_eq!(p.num_qubits(), 12);
}

#[test]
fn problem_type_num_qubits_tsp() {
    let p = ProblemType::TSP {
        num_cities: 3,
        distances: vec![vec![0.0; 3]; 3],
    };
    assert_eq!(p.num_qubits(), 9);
}

#[test]
fn problem_type_num_qubits_custom() {
    let p = ProblemType::Custom {
        num_qubits: 7,
        terms: vec![(vec![0, 1], 1.0)],
    };
    assert_eq!(p.num_qubits(), 7);
}

#[test]
fn problem_type_descriptions() {
    let g = Graph::from_edges(3, &[(0, 1, 1.0)]);
    assert!(ProblemType::MaxCut(g.clone())
        .description()
        .contains("MaxCut"));
    assert!(ProblemType::MinVertexCover(g.clone())
        .description()
        .contains("Vertex Cover"));
    assert!(ProblemType::MaxIndependentSet(g.clone())
        .description()
        .contains("Independent Set"));
    assert!(ProblemType::NumberPartitioning(vec![1.0])
        .description()
        .contains("Partitioning"));
    assert!(ProblemType::GraphColoring(g.clone(), 3)
        .description()
        .contains("Coloring"));
    assert!((ProblemType::MaxKSat {
        num_variables: 3,
        clauses: vec![]
    })
    .description()
    .contains("SAT"));
    assert!((ProblemType::TSP {
        num_cities: 3,
        distances: vec![]
    })
    .description()
    .contains("TSP"));
    assert!((ProblemType::Portfolio {
        assets: 3,
        expected_returns: vec![],
        covariances: vec![],
        risk_factor: 0.5,
        budget: 1
    })
    .description()
    .contains("Portfolio"));
    assert!((ProblemType::Custom {
        num_qubits: 2,
        terms: vec![]
    })
    .description()
    .contains("Custom"));
}

// ============================================================================
// MixerType tests
// ============================================================================

#[test]
fn mixer_type_descriptions() {
    assert!(MixerType::StandardX.description().contains("X"));
    assert!(MixerType::StandardY.description().contains("Y"));
    assert!(MixerType::XY.description().contains("XY"));
    assert!(MixerType::Grover.description().contains("Grover"));
    assert!(MixerType::Ring.description().contains("Ring"));
}

// ============================================================================
// QAOAConfig tests
// ============================================================================

#[test]
fn qaoa_config_default() {
    let cfg = QAOAConfig::default();
    assert_eq!(cfg.depth, 1);
    assert_eq!(cfg.mixer, MixerType::StandardX);
    assert_eq!(cfg.initial_state, InitialState::UniformSuperposition);
    // Standard QAOA applies the mixer in every layer, including the last
    // (issue #40): the default must include the final mixer.
    assert!(cfg.final_mixer);
}

// ============================================================================
// QAOACircuitBuilder tests
// ============================================================================

#[test]
fn builder_new_basic() {
    let g = Graph::cycle(4);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 2);
    assert_eq!(b.num_qubits(), 4);
    assert_eq!(b.num_parameters(), 4);
}

#[test]
fn builder_with_config() {
    let g = Graph::path(3);
    let cfg = QAOAConfig {
        depth: 3,
        mixer: MixerType::StandardY,
        initial_state: InitialState::Zero,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g), cfg);
    assert_eq!(b.num_qubits(), 3);
    assert_eq!(b.num_parameters(), 6);
}

#[test]
fn builder_build_maxcut_produces_circuit() {
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 1);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert_eq!(circuit.num_qubits(), 3);
    assert!(!circuit.gate_counts().is_empty());
}

#[test]
fn builder_build_maxcut_runs_on_simulator() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 1);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    let result = sim.run(&circuit);
    assert!(result.is_ok());
}

#[test]
fn builder_cost_observable_maxcut() {
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 1);
    let obs = b.cost_observable();
    assert!(obs.is_ok());
}

#[test]
fn builder_cost_observable_number_partition() {
    let b = QAOACircuitBuilder::new(
        ProblemType::NumberPartitioning(vec![1.0, 2.0, 3.0]),
        MixerType::StandardX,
        1,
    );
    let obs = b.cost_observable();
    assert!(obs.is_ok());
}

#[test]
fn builder_cost_observable_custom() {
    let b = QAOACircuitBuilder::new(
        ProblemType::Custom {
            num_qubits: 3,
            terms: vec![(vec![0, 1], 1.0), (vec![1, 2], -0.5)],
        },
        MixerType::StandardX,
        1,
    );
    let obs = b.cost_observable();
    assert!(obs.is_ok());
}

#[test]
fn builder_cost_observable_empty_graph_errors() {
    let g = Graph::from_edges(3, &[]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 1);
    assert!(b.cost_observable().is_err());
}

#[test]
fn builder_cost_observable_empty_custom_errors() {
    let b = QAOACircuitBuilder::new(
        ProblemType::Custom {
            num_qubits: 2,
            terms: vec![],
        },
        MixerType::StandardX,
        1,
    );
    assert!(b.cost_observable().is_err());
}

#[test]
fn builder_build_wrong_param_count_errors() {
    let g = Graph::cycle(3);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 2);
    let err = b.build(&[0.5, 0.3]).unwrap_err();
    assert!(err.to_string().contains("expected 4 parameters"), "got: {}", err);
}

#[test]
fn builder_problem_and_mixer_descriptions() {
    let g = Graph::cycle(4);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 1);
    assert!(b.problem_description().contains("MaxCut"));
    assert!(b.mixer_description().contains("X"));
}

// ============================================================================
// Different problem types produce valid circuits
// ============================================================================

#[test]
fn builder_vertex_cover_runs() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MinVertexCover(g), MixerType::StandardX, 1);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

#[test]
fn builder_independent_set_runs() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxIndependentSet(g), MixerType::StandardX, 1);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

#[test]
fn builder_number_partition_runs() {
    let sim = make_sim();
    let b = QAOACircuitBuilder::new(
        ProblemType::NumberPartitioning(vec![1.0, 2.0, 3.0]),
        MixerType::StandardX,
        1,
    );
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

#[test]
fn builder_maxsat_runs() {
    let sim = make_sim();
    let b = QAOACircuitBuilder::new(
        ProblemType::MaxKSat {
            num_variables: 3,
            clauses: vec![(vec![0, 1], vec![false, true], 1.0)],
        },
        MixerType::StandardX,
        1,
    );
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

#[test]
fn builder_custom_problem_runs() {
    let sim = make_sim();
    let b = QAOACircuitBuilder::new(
        ProblemType::Custom {
            num_qubits: 3,
            terms: vec![(vec![0], 1.0), (vec![0, 1], -0.5)],
        },
        MixerType::StandardX,
        1,
    );
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

// ============================================================================
// Different mixer types
// ============================================================================

#[test]
fn builder_standard_y_mixer_runs() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let cfg = QAOAConfig {
        depth: 1,
        mixer: MixerType::StandardY,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g), cfg);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

#[test]
fn builder_xy_mixer_runs() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let cfg = QAOAConfig {
        depth: 1,
        mixer: MixerType::XY,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g), cfg);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

#[test]
fn builder_grover_mixer_fails_loudly() {
    // The Grover mixer is not implemented; requesting it must be an error,
    // not a silently different mixer.
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let cfg = QAOAConfig {
        depth: 1,
        mixer: MixerType::Grover,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g), cfg);
    let err = b.build(&[0.5, 0.3]).unwrap_err();
    assert!(err.to_string().contains("not implemented"), "got: {}", err);
}

#[test]
fn builder_ring_mixer_runs() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let cfg = QAOAConfig {
        depth: 1,
        mixer: MixerType::Ring,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g), cfg);
    let circuit = b.build(&[0.5, 0.3]).unwrap();
    assert!(sim.run(&circuit).is_ok());
}

// ============================================================================
// Initial state variants
// ============================================================================

#[test]
fn builder_zero_initial_state() {
    let sim = make_sim();
    let g = Graph::from_edges(2, &[(0, 1, 1.0)]);
    let cfg = QAOAConfig {
        depth: 1,
        mixer: MixerType::StandardX,
        initial_state: InitialState::Zero,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g), cfg);
    let circuit = b.build(&[0.0, 0.0]).unwrap();
    let result = sim.run(&circuit).unwrap();
    let amps = match &result.state {
        AdaptiveState::Dense(d) => d.amplitudes().to_vec(),
        AdaptiveState::Sparse { state: s, .. } => ferriq_state::DenseState::from_sparse(s)
            .unwrap()
            .amplitudes()
            .to_vec(),
    };
    assert!((amps[0].norm() - 1.0).abs() < 1e-10);
}

// ============================================================================
// Multi-depth QAOA
// ============================================================================

#[test]
fn builder_depth_2_maxcut() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g.clone()), MixerType::StandardX, 2);
    assert_eq!(b.num_parameters(), 4);
    let circuit = b.build(&[0.5, 0.3, 0.4, 0.2]).unwrap();
    assert!(sim.run(&circuit).is_ok());

    let obs = b.cost_observable().unwrap();
    let e = expectation(&sim, &circuit, &obs);
    assert!(e.is_finite());
}

#[test]
fn builder_final_mixer_affects_output() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);

    let cfg_no_final = QAOAConfig {
        depth: 1,
        mixer: MixerType::StandardX,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: false,
    };
    let cfg_with_final = QAOAConfig {
        depth: 1,
        mixer: MixerType::StandardX,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: true,
    };

    let b1 = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g.clone()), cfg_no_final);
    let b2 = QAOACircuitBuilder::with_config(ProblemType::MaxCut(g.clone()), cfg_with_final);

    let obs = b1.cost_observable().unwrap();
    let e1 = expectation(&sim, &b1.build(&[0.5, 0.3]).unwrap(), &obs);
    let e2 = expectation(&sim, &b2.build(&[0.5, 0.3]).unwrap(), &obs);
    assert!((e1 - e2).abs() > 1e-10);
}

// ============================================================================
// Evaluate MaxCut solution
// ============================================================================

#[test]
fn evaluate_maxcut_all_same_partition() {
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
    let cost = evaluate_maxcut_solution(&g, &[true, true, true]);
    assert_eq!(cost, 0.0);
}

#[test]
fn evaluate_maxcut_optimal_bipartition() {
    let g = Graph::from_edges(4, &[(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)]);
    let cost = evaluate_maxcut_solution(&g, &[true, false, false, true]);
    assert_eq!(cost, 4.0);
}

#[test]
fn evaluate_maxcut_weighted() {
    let g = Graph::from_edges(2, &[(0, 1, 3.5)]);
    assert_eq!(evaluate_maxcut_solution(&g, &[true, false]), 3.5);
    assert_eq!(evaluate_maxcut_solution(&g, &[true, true]), 0.0);
}

// ============================================================================
// Evaluate partition solution
// ============================================================================

#[test]
fn evaluate_partition_equal_split() {
    let numbers = vec![1.0, 2.0, 3.0];
    let diff = evaluate_partition_solution(&numbers, &[true, false, true]);
    assert!((diff - 2.0).abs() < 1e-10);
}

#[test]
fn evaluate_partition_perfect_balance() {
    let numbers = vec![5.0, 5.0];
    let diff = evaluate_partition_solution(&numbers, &[true, false]);
    assert!((diff - 0.0).abs() < 1e-10);
}

// ============================================================================
// Random initial parameters
// ============================================================================

#[test]
fn random_params_correct_length() {
    let params = random_initial_parameters(3, Some(42));
    assert_eq!(params.len(), 6);
}

#[test]
fn random_params_deterministic_with_seed() {
    let p1 = random_initial_parameters(2, Some(123));
    let p2 = random_initial_parameters(2, Some(123));
    assert_eq!(p1, p2);
}

#[test]
fn random_params_different_seeds_differ() {
    let p1 = random_initial_parameters(2, Some(1));
    let p2 = random_initial_parameters(2, Some(2));
    assert_ne!(p1, p2);
}

#[test]
fn random_params_in_range() {
    let params = random_initial_parameters(5, Some(0));
    for i in 0..5 {
        let gamma = params[2 * i];
        let beta = params[2 * i + 1];
        assert!((0.0..=std::f64::consts::PI).contains(&gamma));
        assert!((0.0..=std::f64::consts::FRAC_PI_2).contains(&beta));
    }
}

// ============================================================================
// QAOA with gradient integration
// ============================================================================

#[test]
fn qaoa_maxcut_expectation_varies_with_params() {
    let sim = make_sim();
    let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::MaxCut(g), MixerType::StandardX, 1);
    let obs = b.cost_observable().unwrap();

    let cfg = QAOAConfig {
        depth: 1,
        mixer: MixerType::StandardX,
        initial_state: InitialState::UniformSuperposition,
        final_mixer: true,
    };
    let b = QAOACircuitBuilder::with_config(
        ProblemType::MaxCut(Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0)])),
        cfg,
    );

    let e1 = expectation(&sim, &b.build(&[0.1, 0.1]).unwrap(), &obs);
    let e2 = expectation(&sim, &b.build(&[1.0, 1.0]).unwrap(), &obs);
    assert!((e1 - e2).abs() > 1e-6);
}

#[test]
fn qaoa_graph_coloring_fails_loudly() {
    // The graph-coloring encoding is not implemented; historically this
    // silently applied a MaxCut Hamiltonian on the wrong qubits.
    let g = Graph::from_edges(2, &[(0, 1, 1.0)]);
    let b = QAOACircuitBuilder::new(ProblemType::GraphColoring(g, 2), MixerType::StandardX, 1);
    assert_eq!(b.num_qubits(), 4);
    let err = b.build(&[0.5, 0.3]).unwrap_err();
    assert!(err.to_string().contains("not implemented"), "got: {}", err);
    assert!(b.cost_observable().is_err());
}
