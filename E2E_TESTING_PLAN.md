# SimQ End-to-End Testing Plan

## Overview

This document outlines a comprehensive end-to-end testing strategy for SimQ, covering all 8 crates and every major feature. The plan is organized by crate, with cross-crate integration tests and full-pipeline E2E scenarios at the end.

Current state: 168 tests passing, 5 failing (in `simq-state`), 7 integration test files, 39 examples (untested in CI).

---

## 1. simq-core — Type System & Circuit Building

### 1.1 CircuitBuilder (compile-time sized)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 1 | `build_empty_circuit` | Build a circuit with 0 gates | Edge case: empty circuit is valid |
| 2 | `build_single_gate_circuit` | Apply one Hadamard to qubit 0 | Basic gate application |
| 3 | `build_multi_qubit_circuit` | Build circuit with 1–5 qubit sizes | Compile-time const generic sizing |
| 4 | `apply_single_qubit_gates` | Apply H, X, Y, Z, S, T, SX to each qubit | All single-qubit gate types work |
| 5 | `apply_two_qubit_gates` | Apply CNOT, CZ, SWAP, iSWAP, CY, ECR | All two-qubit gate types work |
| 6 | `apply_three_qubit_gates` | Apply Toffoli, Fredkin | Multi-controlled gates |
| 7 | `apply_rotation_gates` | Apply Rx, Ry, Rz, Phase with various angles | Parametric rotation gates |
| 8 | `qubit_index_bounds` | Apply gate to out-of-range qubit | Error handling |
| 9 | `duplicate_qubit_in_gate` | Apply CNOT(q0, q0) | Error: same qubit as control and target |
| 10 | `large_circuit_100_gates` | Build circuit with 100+ gates | Scalability |
| 11 | `circuit_gate_count` | Verify gate_count() matches applied gates | Metadata correctness |
| 12 | `circuit_num_qubits` | Verify num_qubits() returns correct size | Metadata correctness |
| 13 | `circuit_gate_ordering` | Gates appear in insertion order | Deterministic ordering |

### 1.2 DynamicCircuitBuilder (runtime-sized)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 14 | `dynamic_build_n_qubits` | Build circuits with n=1,2,5,10,20 qubits | Runtime sizing |
| 15 | `dynamic_qubit_out_of_range` | Apply gate to qubit >= n | Error handling |
| 16 | `dynamic_matches_static` | Same circuit built both ways produces identical output | Parity between builders |
| 17 | `dynamic_zero_qubits` | Build with n=0 | Edge case handling |
| 18 | `dynamic_large_qubit_count` | Build with n=25+ qubits | Memory scaling |

### 1.3 Parameters & Parametric Circuits

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 19 | `create_parameter` | Create named parameter, verify id and name | Parameter creation |
| 20 | `parameter_registry_add_get` | Add parameters, retrieve by id | Registry CRUD |
| 21 | `parameter_registry_duplicates` | Add same-name parameter twice | Uniqueness enforcement |
| 22 | `bind_parameter_value` | Bind a value and verify it's retrievable | Value binding |
| 23 | `unbind_parameter` | Unbind a previously bound parameter | Value removal |
| 24 | `parametric_circuit_bind_all` | Build parametric circuit, bind all params, execute | Full parametric workflow |
| 25 | `parametric_circuit_partial_bind` | Bind only some params, attempt execution | Error: unbound parameters |
| 26 | `parameter_update_value` | Change bound value and re-execute | Value mutation |

### 1.4 Circuit Visualization

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 27 | `ascii_render_basic` | Render H-CNOT circuit to ASCII | ASCII output is non-empty and formatted |
| 28 | `ascii_render_multi_qubit` | Render 3+ qubit circuit | Multi-wire rendering |
| 29 | `ascii_render_empty` | Render empty circuit | Edge case |
| 30 | `latex_render_basic` | Render circuit to LaTeX | Valid LaTeX output |
| 31 | `latex_render_all_gate_types` | Render circuit with every gate type | Gate symbol coverage |
| 32 | `bloch_sphere_state` | Compute Bloch vector from state | Bloch sphere coordinates |
| 33 | `bloch_sphere_known_states` | |0⟩, |1⟩, |+⟩, |−⟩, |i⟩, |−i⟩ | Known Bloch vectors |

### 1.5 Circuit Debugging

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 34 | `debugger_step_through` | Step through circuit gate by gate | Step-by-step execution |
| 35 | `debugger_state_snapshots` | Capture state after each gate | State correctness at each step |
| 36 | `stateful_debugger_breakpoint` | Set breakpoint, run to breakpoint | Breakpoint mechanism |
| 37 | `debugger_empty_circuit` | Debug empty circuit | Edge case |

### 1.6 Noise Models

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 38 | `depolarizing_channel_identity` | p=0 depolarizing = identity | No-noise baseline |
| 39 | `depolarizing_channel_full` | p=1 depolarizing = maximally mixed | Maximum noise |
| 40 | `amplitude_damping_decay` | Amplitude damping decays |1⟩ to |0⟩ | Physical correctness |
| 41 | `phase_damping_no_amplitude_change` | Phase damping preserves populations | Channel property |
| 42 | `readout_error_flip_probs` | Readout error flips 0↔1 with given probability | Measurement noise |
| 43 | `hardware_noise_model_compose` | Compose gate noise + readout noise | Combined noise model |
| 44 | `gate_timing_validation` | Gate timing values are non-negative | Constraint validation |
| 45 | `noise_model_serialization` | Serialize/deserialize noise model | Roundtrip fidelity |

### 1.7 Validation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 46 | `validate_well_formed_circuit` | Valid circuit passes all rules | Positive case |
| 47 | `validate_qubit_reuse_error` | Same qubit used as both control and target | Rule violation |
| 48 | `validate_dag_acyclicity` | DAG has no cycles | DAG property |
| 49 | `validate_dag_dependencies` | Gates on same qubit are ordered | Dependency tracking |
| 50 | `validation_report_format` | Report contains rule name, severity, message | Report structure |
| 51 | `validate_large_circuit` | Validate 1000-gate circuit in <100ms | Performance bound |

### 1.8 Serialization

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 52 | `json_roundtrip_simple` | Serialize/deserialize simple circuit to JSON | JSON fidelity |
| 53 | `json_roundtrip_all_gates` | Circuit with every gate type | Full gate coverage |
| 54 | `json_roundtrip_parametric` | Parametric circuit with bound/unbound params | Parameter preservation |
| 55 | `binary_roundtrip` | Serialize/deserialize to binary (bincode) | Binary format |
| 56 | `postcard_roundtrip` | Serialize/deserialize with postcard | Embedded format |
| 57 | `cross_format_equivalence` | JSON and binary produce equivalent circuits | Format independence |
| 58 | `deserialization_invalid_json` | Malformed JSON input | Error handling |
| 59 | `serialization_cache_hit` | Serialize same circuit twice, second is cached | Cache effectiveness |

---

## 2. simq-gates — Gate Library & Caching

### 2.1 Standard Gates — Mathematical Correctness

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 60 | `hadamard_matrix` | H matrix = (1/√2)[[1,1],[1,-1]] | Matrix values |
| 61 | `pauli_x_matrix` | X matrix = [[0,1],[1,0]] | Bit flip |
| 62 | `pauli_y_matrix` | Y = [[0,-i],[i,0]] | Phase + bit flip |
| 63 | `pauli_z_matrix` | Z = [[1,0],[0,-1]] | Phase flip |
| 64 | `identity_matrix` | I = [[1,0],[0,1]] | Identity |
| 65 | `s_gate_matrix` | S = [[1,0],[0,i]] | π/2 phase |
| 66 | `t_gate_matrix` | T = [[1,0],[0,e^(iπ/4)]] | π/4 phase |
| 67 | `sx_gate_matrix` | SX = √X | Square root of X |
| 68 | `cnot_matrix` | CNOT 4x4 matrix | Controlled-NOT |
| 69 | `cz_matrix` | CZ 4x4 matrix | Controlled-Z |
| 70 | `swap_matrix` | SWAP 4x4 matrix | Qubit swap |
| 71 | `iswap_matrix` | iSWAP 4x4 matrix | iSWAP gate |
| 72 | `toffoli_matrix` | Toffoli 8x8 matrix | 3-qubit controlled |
| 73 | `all_gates_unitary` | Every standard gate's matrix is unitary (U†U = I) | Unitarity |
| 74 | `rotation_x_angles` | Rx(0)=I, Rx(π)=iX, Rx(2π)=-I | Rotation correctness |
| 75 | `rotation_y_angles` | Ry(0)=I, Ry(π)=iY | Rotation correctness |
| 76 | `rotation_z_angles` | Rz(0)=I, Rz(π)=iZ | Rotation correctness |
| 77 | `phase_gate_angles` | Phase(0)=I, Phase(π)=Z, Phase(π/2)=S | Phase gate |
| 78 | `gate_inverse_property` | G * G† = I for all gates | Inverse correctness |
| 79 | `gate_num_qubits` | Each gate reports correct qubit count | Metadata |

### 2.2 Compile-Time Cache — Multi-Level

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 80 | `cache_level1_common_angles` | π/4, π/2, π hit Level 1 cache | Common angle lookup |
| 81 | `cache_level2_clifford_t` | π/8, π/16, π/32 hit Level 2 | Clifford+T lookup |
| 82 | `cache_level3_pi_fractions` | π/3, π/5, π/6 hit Level 3 | Pi fraction lookup |
| 83 | `cache_level4_vqe_range` | Angles in [0, π/4] with 256 steps | VQE range lookup |
| 84 | `cache_level5_qaoa_range` | Angles in [0, π] with 100 steps | QAOA range lookup |
| 85 | `cache_level6_fallback` | Arbitrary angle falls through to runtime | Fallback computation |
| 86 | `cache_accuracy_vs_runtime` | Cached matrices match runtime-computed within ε | Numerical accuracy |
| 87 | `cache_all_rx_angles` | Rx at every cached angle matches direct computation | Rx cache correctness |
| 88 | `cache_all_ry_angles` | Same for Ry | Ry cache correctness |
| 89 | `cache_all_rz_angles` | Same for Rz | Rz cache correctness |
| 90 | `cache_memory_footprint` | Total cache size ≈ 70KB | Memory budget |

### 2.3 Custom Gates

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 91 | `custom_gate_2x2` | Build custom 1-qubit gate from matrix | Custom gate creation |
| 92 | `custom_gate_4x4` | Build custom 2-qubit gate | Multi-qubit custom gate |
| 93 | `custom_gate_non_unitary_rejected` | Non-unitary matrix is rejected | Validation |
| 94 | `custom_gate_with_description` | Gate has name and description | Metadata |
| 95 | `parametric_custom_gate` | Parametric custom gate with angle parameter | Parametric support |
| 96 | `parametric_custom_gate_bind` | Bind parameter and evaluate matrix | Parameter binding |
| 97 | `gate_registry_register_retrieve` | Register and retrieve custom gate | Registry CRUD |
| 98 | `gate_registry_overwrite` | Re-register same name | Overwrite behavior |
| 99 | `gate_registry_list_all` | List all registered gates | Enumeration |

### 2.4 Matrix Operations

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 100 | `matrix_multiply_2x2` | Multiply two 2x2 complex matrices | Basic multiplication |
| 101 | `matrix_multiply_4x4` | Multiply two 4x4 matrices | Larger multiplication |
| 102 | `tensor_product_2x2` | Tensor product of two 2x2 → 4x4 | Tensor product |
| 103 | `is_unitary_true` | Unitary matrix returns true | Unitarity check |
| 104 | `is_unitary_false` | Non-unitary matrix returns false | Negative case |
| 105 | `is_hermitian_true` | Hermitian matrix returns true | Hermiticity check |
| 106 | `is_hermitian_false` | Non-Hermitian matrix returns false | Negative case |
| 107 | `matrix_multiply_identity` | A * I = A | Identity property |

---

## 3. simq-state — Quantum State Representations

### 3.1 SparseState

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 108 | `sparse_create_zero_state` | Create |000...0⟩ sparse state | Initialization |
| 109 | `sparse_set_amplitude` | Set amplitude at basis index | Amplitude mutation |
| 110 | `sparse_get_amplitude` | Get amplitude at set/unset indices | Retrieval |
| 111 | `sparse_normalization` | Normalize state to unit norm | Normalization |
| 112 | `sparse_probability` | Probabilities sum to 1.0 | Probability conservation |
| 113 | `sparse_num_nonzero` | Count of non-zero amplitudes | Sparsity tracking |
| 114 | `sparse_apply_gate_h` | Apply Hadamard to sparse state | Gate application |
| 115 | `sparse_large_qubit_count` | 30+ qubit sparse state with few non-zero | Scalability |

### 3.2 DenseState

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 116 | `dense_create_zero_state` | Create |000...0⟩ dense state | Initialization |
| 117 | `dense_alignment` | State vector is 64-byte aligned | SIMD alignment |
| 118 | `dense_apply_gate_h` | Apply Hadamard | Gate application |
| 119 | `dense_apply_gate_cnot` | Apply CNOT | Two-qubit gate |
| 120 | `dense_probabilities` | All probabilities sum to 1.0 | Norm conservation |
| 121 | `dense_state_equality` | Two identical states are equal | Equality |
| 122 | `dense_clone_independence` | Cloned state is independent | Clone correctness |

### 3.3 AdaptiveState

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 123 | `adaptive_starts_sparse` | New adaptive state starts as sparse | Default representation |
| 124 | `adaptive_converts_to_dense` | After enough non-zero amplitudes, converts to dense | Threshold conversion |
| 125 | `adaptive_threshold_config` | Custom density threshold works | Configuration |
| 126 | `adaptive_gate_application` | Gates work across conversions | Correctness through conversion |
| 127 | `adaptive_matches_dense` | Adaptive result matches pure dense for same circuit | Result equivalence |

### 3.4 CowState (Copy-on-Write)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 128 | `cow_share_state` | Two CowStates share underlying data | Shared reference |
| 129 | `cow_copy_on_write` | Mutation creates independent copy | CoW semantics |
| 130 | `cow_no_copy_on_read` | Reading doesn't trigger copy | Read optimization |
| 131 | `cow_branching` | Fork state, apply different gates, compare | Branching workflow |

### 3.5 DensityMatrix

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 132 | `density_matrix_pure_state` | |ψ⟩⟨ψ| for pure state | Pure state representation |
| 133 | `density_matrix_mixed_state` | Classical mixture of |0⟩ and |1⟩ | Mixed state |
| 134 | `density_matrix_trace_one` | Tr(ρ) = 1 | Trace preservation |
| 135 | `density_matrix_positive_semidefinite` | All eigenvalues ≥ 0 | Physical validity |
| 136 | `density_matrix_purity` | Tr(ρ²) = 1 for pure, < 1 for mixed | Purity metric |
| 137 | `density_matrix_gate_evolution` | UρU† for unitary gate | Gate application |
| 138 | `density_matrix_partial_trace` | Trace out subsystem | Partial trace |

### 3.6 Measurement

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 139 | `measure_zero_state` | Measure |0⟩ → always 0 | Deterministic outcome |
| 140 | `measure_one_state` | Measure |1⟩ → always 1 | Deterministic outcome |
| 141 | `measure_superposition` | Measure |+⟩ → ~50/50 (statistical) | Probabilistic outcome |
| 142 | `measure_bell_state` | Measure Bell state → correlated outcomes | Entanglement |
| 143 | `measure_collapse` | Post-measurement state is collapsed | State collapse |
| 144 | `measure_partial` | Measure one qubit of multi-qubit state | Partial measurement |
| 145 | `batch_sampling_distribution` | 10000 shots → distribution matches probabilities within χ² | Statistical correctness |
| 146 | `computational_basis_all_outcomes` | All 2^n outcomes appear for uniform state | Coverage |

### 3.7 Observables

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 147 | `pauli_x_expectation` | ⟨+|X|+⟩ = 1, ⟨0|X|0⟩ = 0 | X observable |
| 148 | `pauli_z_expectation` | ⟨0|Z|0⟩ = 1, ⟨1|Z|1⟩ = -1 | Z observable |
| 149 | `pauli_string_expectation` | ⟨ψ|Z⊗Z|ψ⟩ for Bell state | Multi-qubit observable |
| 150 | `observable_linearity` | ⟨ψ|aA+bB|ψ⟩ = a⟨A⟩ + b⟨B⟩ | Linearity |
| 151 | `observable_hermitian_real` | Expectation values are real | Hermitian property |

### 3.8 SIMD Kernels

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 152 | `simd_single_qubit_matches_scalar` | SIMD gate application matches scalar fallback | Correctness |
| 153 | `simd_two_qubit_matches_scalar` | Same for 2-qubit gates | Correctness |
| 154 | `simd_controlled_gate_matches_scalar` | Same for controlled gates | Correctness |
| 155 | `simd_diagonal_gate_matches_scalar` | Same for diagonal gates | Correctness |
| 156 | `simd_norm_computation` | SIMD norm matches scalar norm | Norm correctness |
| 157 | `simd_all_qubit_positions` | Gate on qubit 0,1,...,n-1 all correct | Qubit indexing |

### 3.9 Monte Carlo Simulator

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 158 | `mc_no_noise_matches_ideal` | MC with no noise = ideal simulation | Baseline |
| 159 | `mc_depolarizing_reduces_fidelity` | Depolarizing noise reduces state fidelity | Noise effect |
| 160 | `mc_trajectory_count` | Correct number of trajectories generated | Configuration |
| 161 | `mc_statistical_convergence` | More trajectories → tighter confidence interval | Convergence |

### 3.10 Density Matrix Simulator

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 162 | `dms_pure_state_evolution` | Matches state vector simulator for noiseless | Equivalence |
| 163 | `dms_noisy_evolution` | Noise channels reduce purity | Noise effect |
| 164 | `dms_channel_composition` | Multiple noise channels compose correctly | Composition |

---

## 4. simq-compiler — Circuit Optimization

### 4.1 Optimization Levels

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 165 | `o0_no_optimization` | O0 produces unchanged circuit | Identity transform |
| 166 | `o1_basic_optimization` | O1 removes identity gates | Basic cleanup |
| 167 | `o2_standard_optimization` | O2 fuses adjacent gates | Gate fusion |
| 168 | `o3_aggressive_optimization` | O3 applies all passes | Full optimization |
| 169 | `optimization_preserves_semantics` | For each level: optimized circuit produces same measurement distribution | Semantic preservation |

### 4.2 Gate Fusion

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 170 | `fuse_adjacent_single_qubit` | H followed by H = I (removed) | Single-qubit fusion |
| 171 | `fuse_rotations` | Rz(a) followed by Rz(b) = Rz(a+b) | Rotation merging |
| 172 | `no_fuse_different_qubits` | Gates on different qubits not fused | Non-adjacent gates |
| 173 | `fuse_three_gates` | X-Y-Z sequence fused | Multi-gate fusion |
| 174 | `fusion_reduces_gate_count` | Post-fusion gate count < pre-fusion | Gate reduction |

### 4.3 Dead Code Elimination

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 175 | `dce_remove_identity` | I gates removed | Identity elimination |
| 176 | `dce_remove_cancel_pairs` | X-X pair removed | Inverse cancellation |
| 177 | `dce_no_false_removal` | Non-cancelling gates preserved | Correctness |

### 4.4 Gate Commutation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 178 | `commute_z_gates` | Z-type gates commute through controls | Z commutation |
| 179 | `commute_x_gates` | X-type gates commute through targets | X commutation |
| 180 | `commutation_enables_fusion` | Commutation followed by fusion reduces gate count | Combined optimization |
| 181 | `non_commuting_preserved` | Non-commuting gates maintain order | Correctness |

### 4.5 Template Matching & Substitution

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 182 | `template_match_cnot_h` | Known template patterns detected | Pattern detection |
| 183 | `template_substitution` | Matched patterns replaced with optimized equivalent | Substitution |
| 184 | `template_no_false_match` | Non-matching patterns not substituted | False positive avoidance |

### 4.6 Gate Decomposition

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 185 | `decompose_toffoli` | Toffoli → CNOT+single-qubit gates | 3-qubit decomposition |
| 186 | `decompose_to_basis_set` | Arbitrary gate → {CNOT, Rz, SX} basis | Basis decomposition |
| 187 | `decompose_clifford_t` | Gates decomposed to Clifford+T set | Fault-tolerant decomposition |
| 188 | `decomposition_preserves_unitary` | Product of decomposed gates = original unitary | Mathematical correctness |
| 189 | `decompose_two_qubit` | Arbitrary 2-qubit unitary decomposed | KAK decomposition |
| 190 | `decompose_single_qubit` | Arbitrary 1-qubit unitary → ZYZ | Euler decomposition |

### 4.7 Lazy Evaluation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 191 | `lazy_deferred_computation` | Matrix not computed until needed | Deferral |
| 192 | `lazy_force_evaluation` | Forced evaluation produces correct result | Correctness |
| 193 | `lazy_cache_after_eval` | Second evaluation reuses cached result | Caching |
| 194 | `lazy_with_parameters` | Lazy parametric gates evaluated on bind | Parametric lazy |

### 4.8 Circuit Analysis

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 195 | `analysis_gate_count` | Correct gate count by type | Gate statistics |
| 196 | `analysis_circuit_depth` | Correct circuit depth (critical path) | Depth calculation |
| 197 | `analysis_parallelism` | Identify parallelizable gate layers | Parallelism metric |
| 198 | `resource_estimate` | Qubit count, gate count, depth, T-count | Resource estimation |

### 4.9 Execution Planning

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 199 | `execution_plan_ordering` | Execution plan respects gate dependencies | Ordering correctness |
| 200 | `execution_plan_parallelism` | Independent gates scheduled in parallel | Parallel scheduling |
| 201 | `execution_plan_memory` | Memory estimate is reasonable | Resource tracking |

### 4.10 Adaptive & Cached Compilation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 202 | `adaptive_selects_passes` | Adaptive compiler selects passes based on circuit | Adaptivity |
| 203 | `cached_compiler_hit` | Same circuit compiled twice → cache hit | Cache effectiveness |
| 204 | `cached_compiler_invalidation` | Modified circuit → cache miss | Cache correctness |
| 205 | `circuit_fingerprint_uniqueness` | Different circuits produce different fingerprints | Fingerprint quality |

### 4.11 Hardware-Aware Compilation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 206 | `hw_ibm_basis_gates` | Compiled to IBM native gate set | Hardware targeting |
| 207 | `hw_google_basis_gates` | Compiled to Google native gate set | Hardware targeting |
| 208 | `hw_connectivity_respected` | Gates only on connected qubits | Topology constraint |
| 209 | `hw_swap_insertion` | SWAPs inserted for non-adjacent qubits | Routing |

---

## 5. simq-sim — Simulator Engine

### 5.1 Core Simulation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 210 | `sim_empty_circuit` | Simulate empty circuit → |0...0⟩ | Baseline |
| 211 | `sim_hadamard` | H|0⟩ = |+⟩ | Basic simulation |
| 212 | `sim_bell_state` | H-CNOT creates Bell state | Entanglement creation |
| 213 | `sim_ghz_state` | GHZ state on 3+ qubits | Multi-qubit entanglement |
| 214 | `sim_teleportation` | Quantum teleportation protocol | Multi-step algorithm |
| 215 | `sim_all_single_qubit_gates` | Each single-qubit gate produces correct state | Gate coverage |
| 216 | `sim_all_two_qubit_gates` | Each two-qubit gate produces correct state | Gate coverage |
| 217 | `sim_measurement_shots` | Simulate with 1, 100, 10000 shots | Shot count |
| 218 | `sim_deterministic_seed` | Same seed → same results | Reproducibility |
| 219 | `sim_different_seeds` | Different seeds → different results (statistically) | Randomness |

### 5.2 Simulator Configuration

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 220 | `config_sparse_threshold` | Custom sparse threshold affects state representation | Threshold config |
| 221 | `config_parallel_threshold` | Parallel threshold affects execution strategy | Parallelism config |
| 222 | `config_default` | Default config produces valid results | Defaults |
| 223 | `config_shots_zero` | Zero shots → error or empty result | Edge case |

### 5.3 Execution Statistics

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 224 | `stats_gate_applications` | Correct number of gate applications tracked | Gate counting |
| 225 | `stats_execution_time` | Execution time is positive and reasonable | Timing |
| 226 | `stats_state_conversions` | Sparse↔Dense conversions tracked | Conversion tracking |

### 5.4 Gradient Computation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 227 | `parameter_shift_single` | Gradient of Ry(θ) at θ=0 matches analytical | Single parameter |
| 228 | `parameter_shift_multi` | Gradient of multi-parameter circuit | Multi-parameter |
| 229 | `parameter_shift_accuracy` | Gradients match finite difference within ε | Numerical accuracy |
| 230 | `finite_difference_gradient` | FD gradient matches analytical | Finite difference method |
| 231 | `autodiff_gradient` | AD gradient matches parameter shift | Auto-differentiation |
| 232 | `gradient_batch_evaluation` | Batch gradient = sequential gradients | Batch correctness |
| 233 | `gradient_zero_at_extremum` | Gradient ≈ 0 at known minimum | Extremum detection |

### 5.5 Classical Optimizers

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 234 | `optimizer_lbfgs_quadratic` | L-BFGS minimizes f(x)=x² | Basic optimization |
| 235 | `optimizer_nelder_mead` | Nelder-Mead finds minimum | Simplex method |
| 236 | `optimizer_adam` | Adam converges on test function | Adaptive learning rate |
| 237 | `optimizer_momentum` | Momentum SGD converges | Momentum method |
| 238 | `optimizer_convergence_criteria` | Optimization stops at convergence | Termination |
| 239 | `optimizer_max_iterations` | Optimization stops at max iterations | Bounds |

### 5.6 VQE (Variational Quantum Eigensolver)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 240 | `vqe_h2_ground_state` | VQE finds H₂ ground state energy within chemical accuracy | Physical correctness |
| 241 | `vqe_hardware_efficient_ansatz` | HEA circuit is well-formed | Ansatz generation |
| 242 | `vqe_energy_decreases` | Energy decreases over iterations | Optimization progress |
| 243 | `vqe_parameter_count` | Correct number of variational parameters | Parameter counting |
| 244 | `vqe_convergence_monitoring` | Convergence monitor tracks energy history | Monitoring |

### 5.7 QAOA (Quantum Approximate Optimization Algorithm)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 245 | `qaoa_maxcut_triangle` | QAOA finds MaxCut of 3-node triangle | Known solution |
| 246 | `qaoa_maxcut_square` | QAOA on 4-node cycle | Larger instance |
| 247 | `qaoa_circuit_structure` | p layers of mixer+problem Hamiltonians | Circuit structure |
| 248 | `qaoa_all_graph_types` | Complete, cycle, path, star, grid graphs | Graph coverage |
| 249 | `qaoa_all_problems` | MaxCut, number partitioning, graph coloring, TSP | Problem coverage |
| 250 | `qaoa_all_mixers` | Standard X, XY, Grover mixers | Mixer coverage |
| 251 | `qaoa_p_layers` | p=1,2,3,5 layers | Depth scaling |
| 252 | `qaoa_observable_generation` | Problem Hamiltonian → Pauli observables | Observable conversion |
| 253 | `qaoa_solution_quality` | Solution quality improves with p | Approximation ratio |

---

## 6. simq-backend — Hardware Abstraction

### 6.1 Backend Trait & Selection

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 254 | `backend_local_simulator` | Create and use local simulator backend | Local backend |
| 255 | `backend_capabilities_query` | Query supported gates, qubit count, connectivity | Capability reporting |
| 256 | `backend_selector_criteria` | Select backend by criteria (speed, accuracy, cost) | Selection logic |
| 257 | `backend_selector_fallback` | Fallback when preferred backend unavailable | Fallback behavior |

### 6.2 Transpilation

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 258 | `transpile_to_ibm_basis` | Circuit transpiled to {CNOT, ID, RZ, SX, X} | IBM basis |
| 259 | `transpile_preserves_semantics` | Transpiled circuit same unitary as original | Semantic preservation |
| 260 | `transpile_custom_basis` | Transpile to arbitrary basis gate set | Custom basis |
| 261 | `transpile_already_native` | Circuit already in basis → no changes | Identity transpilation |

### 6.3 Qubit Routing (SABRE)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 262 | `routing_linear_topology` | Route on linear qubit connectivity | Linear layout |
| 263 | `routing_grid_topology` | Route on 2D grid connectivity | Grid layout |
| 264 | `routing_all_to_all` | All-to-all connectivity → no SWAPs needed | Trivial routing |
| 265 | `routing_swap_count` | SWAP count is reasonable (not excessive) | Routing quality |
| 266 | `routing_preserves_semantics` | Routed circuit equivalent to original | Semantic preservation |
| 267 | `sabre_deterministic` | Same input → same routing | Determinism |

### 6.4 IBM Quantum Backend (Mock/Offline)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 268 | `ibm_backend_creation` | Create IBM backend with config | Instantiation |
| 269 | `ibm_gate_set` | IBM backend reports correct native gates | Gate set |
| 270 | `ibm_topology` | IBM backend reports correct qubit connectivity | Topology |
| 271 | `ibm_transpile_and_route` | Full transpilation pipeline for IBM | End-to-end |

---

## 7. simq-macros — Procedural Macros

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 272 | `macro_circuit_builder` | Macro generates correct circuit builder code | Code generation |
| 273 | `macro_compile_error` | Invalid macro input produces compile error | Error messages |

---

## 8. simq (Main Crate) — Re-exports & Integration

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 274 | `reexport_all_public_types` | All documented public types accessible via `simq::` | Re-export completeness |
| 275 | `version_consistency` | All crate versions are consistent | Version management |

---

## 9. Cross-Crate Integration Tests

These tests verify that the crates work together correctly across boundaries.

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 276 | `build_compile_simulate` | Build circuit → compile → simulate → measure | Core pipeline |
| 277 | `build_serialize_deserialize_simulate` | Build → serialize → deserialize → simulate | Serialization in pipeline |
| 278 | `parametric_gradient_optimize` | Parametric circuit → gradient → optimizer step | Variational pipeline |
| 279 | `custom_gate_in_circuit` | Register custom gate → use in circuit → simulate | Custom gate pipeline |
| 280 | `noise_model_monte_carlo` | Circuit + noise model → MC simulation → statistics | Noisy simulation pipeline |
| 281 | `density_matrix_noisy_sim` | Circuit + noise → density matrix simulation | DM simulation pipeline |
| 282 | `hardware_aware_full_pipeline` | Build → HW compile → transpile → route → simulate | Hardware pipeline |
| 283 | `cache_hit_across_simulations` | Run same circuit twice → compiler cache hit | Cross-run caching |
| 284 | `adaptive_state_full_circuit` | Large circuit with adaptive state Sparse→Dense | Adaptive state pipeline |
| 285 | `cow_state_branching_compare` | Fork state → apply different gates → compare results | CoW branching pipeline |

---

## 10. End-to-End Scenario Tests

These tests simulate realistic user workflows from start to finish.

### 10.1 Quantum Algorithm Scenarios

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 286 | `e2e_quantum_teleportation` | Full teleportation: prepare, entangle, measure, correct | Teleportation protocol |
| 287 | `e2e_superdense_coding` | Send 2 classical bits via 1 qubit | Communication protocol |
| 288 | `e2e_deutsch_jozsa` | Distinguish constant vs balanced oracle | DJ algorithm |
| 289 | `e2e_bernstein_vazirani` | Find hidden string s | BV algorithm |
| 290 | `e2e_grover_search_2qubit` | Search 4-element database | Grover's on 2 qubits |
| 291 | `e2e_grover_search_3qubit` | Search 8-element database | Grover's on 3 qubits |
| 292 | `e2e_qft_3qubit` | 3-qubit Quantum Fourier Transform | QFT |
| 293 | `e2e_phase_estimation` | Estimate eigenvalue of unitary | QPE |
| 294 | `e2e_vqe_h2_full` | Full VQE: ansatz → optimize → converge → report energy | VQE workflow |
| 295 | `e2e_qaoa_maxcut_full` | Full QAOA: graph → problem → optimize → solution | QAOA workflow |

### 10.2 Workflow Scenarios

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 296 | `e2e_circuit_design_iterate` | Build → visualize → modify → recompile → simulate | Iterative design |
| 297 | `e2e_noise_characterization` | Ideal vs noisy simulation → fidelity comparison | Noise analysis |
| 298 | `e2e_optimizer_comparison` | Same VQE with LBFGS vs Adam vs NM → compare convergence | Optimizer benchmark |
| 299 | `e2e_circuit_optimization_comparison` | Same circuit at O0/O1/O2/O3 → compare depth and fidelity | Optimization levels |
| 300 | `e2e_hardware_targeting` | Same circuit → IBM pipeline vs Google pipeline | Cross-hardware |
| 301 | `e2e_large_circuit_20q` | 20-qubit circuit end-to-end | Scalability |
| 302 | `e2e_parametric_sweep` | Sweep parameter from 0→2π, plot expectation value | Parameter sweep |
| 303 | `e2e_serialization_roundtrip_full` | Build → optimize → serialize → load → simulate → compare | Full serialization |

### 10.3 Error Handling & Edge Cases

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 304 | `e2e_invalid_circuit_rejected` | Invalid circuits caught at each stage | Error propagation |
| 305 | `e2e_unbound_parameters_error` | Simulate with unbound parameters → clear error | Error message quality |
| 306 | `e2e_zero_shot_measurement` | 0 shots → appropriate behavior | Edge case |
| 307 | `e2e_single_qubit_all_gates` | Every gate type on a 1-qubit circuit | Minimum circuit |
| 308 | `e2e_max_qubit_stress` | Push to maximum feasible qubit count | Resource limits |

### 10.4 Performance Regression Tests

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 309 | `perf_bell_state_under_1ms` | Bell state simulation completes in < 1ms | Speed floor |
| 310 | `perf_10q_circuit_under_100ms` | 10-qubit, 100-gate circuit under 100ms | Mid-scale performance |
| 311 | `perf_cache_lookup_under_10ns` | Cached angle lookup under 10ns (benchmark) | Cache speed |
| 312 | `perf_compilation_o2_under_50ms` | O2 compilation of 100-gate circuit under 50ms | Compilation speed |
| 313 | `perf_memory_20q_under_512mb` | 20-qubit simulation uses < 512MB | Memory budget |

---

## 11. Property-Based Tests (using `proptest`)

| # | Test | Description | Validates |
|---|------|-------------|-----------|
| 314 | `prop_any_unitary_preserves_norm` | Random unitary gate preserves state norm | Unitarity invariant |
| 315 | `prop_measurement_probs_sum_to_one` | For any state, measurement probabilities sum to 1 | Probability conservation |
| 316 | `prop_serialization_roundtrip` | Any circuit survives JSON roundtrip | Serialization completeness |
| 317 | `prop_optimization_preserves_unitary` | Random circuit: optimized unitary = original unitary | Optimization correctness |
| 318 | `prop_gate_inverse` | For any rotation angle, G(θ)G(-θ) = I | Inverse property |
| 319 | `prop_transpile_preserves_unitary` | Transpiled circuit has same unitary | Transpilation correctness |
| 320 | `prop_commuting_gates_reorder` | Commuting gates produce same result in any order | Commutation correctness |

---

## 12. Existing Test Fixes Required

The following 5 tests are currently failing and should be fixed before new tests are added:

| Crate | Test | Failure |
|-------|------|---------|
| `simq-state` | `dense_state::tests::test_get_all_probabilities` | Assertion failure |
| `simq-state` | `measurement::tests::test_batch_sampling` | Assertion failure |
| `simq-state` | `measurement::tests::test_computational_basis_measurement` | Assertion failure |
| `simq-state` | `simd::kernels::tests::test_norm_avx2` | Norm = 0.0, expected 2.0 |
| `simq-state` | `simd::kernels::tests::test_norm_sse2` | Norm = 0.0, expected 2.0 |

These failures likely stem from a SIMD kernel bug in norm computation, which cascades to probability and measurement tests.

---

## 13. Test Infrastructure Recommendations

### 13.1 Test Organization
```
tests/
├── e2e/
│   ├── algorithms.rs          # Tests 286-295
│   ├── workflows.rs           # Tests 296-303
│   ├── error_handling.rs      # Tests 304-308
│   └── performance.rs         # Tests 309-313
├── integration/
│   ├── cross_crate.rs         # Tests 276-285
│   └── property_based.rs      # Tests 314-320
└── each crate keeps its own unit + integration tests
```

### 13.2 Test Utilities
- **`test_helpers` module**: Common circuit builders (Bell, GHZ, QFT), state assertions (amplitude within ε, probability within δ), statistical tests (χ² for measurement distributions)
- **`assert_approx_eq!` macro**: Compare Complex64 within tolerance (1e-10 for exact, 1e-6 for numerical)
- **`assert_unitary!` macro**: Verify matrix unitarity
- **`assert_probability_distribution!` macro**: χ² test for measurement results vs expected distribution

### 13.3 CI Configuration
- Run unit tests on every PR
- Run integration tests on every PR
- Run E2E scenario tests on merge to main
- Run performance tests nightly with regression alerts
- Run property-based tests nightly (more iterations than PR runs)

### 13.4 Coverage Targets
- **Unit test coverage**: > 85% line coverage per crate
- **Integration coverage**: Every public API function called at least once
- **E2E coverage**: Every example in `/examples/` has a corresponding test

---

## 14. Priority & Phasing

### Phase 1 — Fix & Foundation (Week 1)
1. Fix the 5 failing SIMD/measurement tests
2. Implement test utilities (helpers, macros)
3. Tests 60-79 (gate mathematical correctness — these are the foundation)
4. Tests 108-122 (state representation basics)

### Phase 2 — Core Pipeline (Week 2)
5. Tests 1-18 (circuit builder)
6. Tests 139-146 (measurement)
7. Tests 210-219 (core simulation)
8. Tests 276-278 (cross-crate pipeline: build→compile→simulate)

### Phase 3 — Advanced Features (Week 3)
9. Tests 80-90 (caching system)
10. Tests 165-174 (compiler optimization)
11. Tests 185-190 (gate decomposition)
12. Tests 227-233 (gradient computation)

### Phase 4 — Algorithms & Hardware (Week 4)
13. Tests 240-253 (VQE & QAOA)
14. Tests 254-271 (backend & routing)
15. Tests 286-295 (E2E algorithm scenarios)

### Phase 5 — Robustness & Performance (Week 5)
16. Tests 304-308 (error handling & edge cases)
17. Tests 309-313 (performance regression)
18. Tests 314-320 (property-based tests)
19. Tests 296-303 (workflow scenarios)

---

## Summary

| Category | Test Count |
|----------|-----------|
| simq-core (circuit, params, viz, debug, noise, validation, serialization) | 59 |
| simq-gates (standard, cache, custom, matrix ops) | 48 |
| simq-state (sparse, dense, adaptive, cow, density, measurement, observable, SIMD, MC, DM sim) | 57 |
| simq-compiler (optimization, fusion, DCE, commutation, templates, decomposition, lazy, analysis, planning, cache, HW) | 45 |
| simq-sim (core, config, stats, gradient, optimizers, VQE, QAOA) | 44 |
| simq-backend (backend, transpilation, routing, IBM) | 18 |
| simq-macros | 2 |
| simq (main crate) | 2 |
| Cross-crate integration | 10 |
| E2E scenarios | 23 |
| Property-based | 7 |
| **Total** | **320** |
