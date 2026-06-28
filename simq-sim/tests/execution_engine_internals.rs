use num_complex::Complex64;
use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use simq_sim::execution_engine::adaptive::AdaptiveStrategy;
use simq_sim::execution_engine::cache::{
    CachedMatrix, GateCacheKey, GateMatrixCache, OrderedFloat,
};
use simq_sim::execution_engine::checkpoint::CheckpointManager;
use simq_sim::execution_engine::config::{ExecutionConfig, ExecutionMode, ParallelStrategy};
use simq_sim::execution_engine::recovery::RecoveryPolicy;
use simq_sim::execution_engine::telemetry::{ExecutionMetrics, ExecutionTelemetry};
use simq_sim::execution_engine::validation::{
    validate_finite, validate_normalization, validate_state,
};
use simq_sim::execution_engine::ExecutionEngine;
use simq_state::{AdaptiveState, DenseState, SparseState};
use std::sync::Arc;
use std::time::Duration;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// ============================================================================
// AdaptiveStrategy
// ============================================================================

#[test]
fn adaptive_strategy_default() {
    let s = AdaptiveStrategy::default();
    let dense = DenseState::new(1).unwrap();
    let state = AdaptiveState::Dense(dense);
    // Dense state is already dense, so convert_to_dense checks density > threshold
    // A 1-qubit dense state has density 1.0 which is > 0.1 threshold
    assert!(s.should_convert_to_dense(&state));
    assert!(!s.should_parallelize_gate(&state));
}

#[test]
fn adaptive_strategy_dense_conversion_thresholds() {
    let s = AdaptiveStrategy::new(0.5, 1 << 20, 1 << 10);
    let sparse = SparseState::new(10).unwrap();
    let state = AdaptiveState::Sparse {
        state: sparse,
        threshold: 0.5,
    };
    assert!(!s.should_convert_to_dense(&state));
}

#[test]
fn adaptive_strategy_select_sequential_mode() {
    let s = AdaptiveStrategy::new(0.1, 1 << 20, 1 << 20);
    let dense = DenseState::new(2).unwrap();
    let state = AdaptiveState::Dense(dense);
    let mode = s.select_execution_mode(&state, 5);
    assert_eq!(mode, ExecutionMode::Sequential);
}

#[test]
fn adaptive_strategy_select_parallel_for_many_gates() {
    let s = AdaptiveStrategy::new(0.1, 1 << 20, 1 << 10);
    let dense = DenseState::new(2).unwrap();
    let state = AdaptiveState::Dense(dense);
    let mode = s.select_execution_mode(&state, 200);
    assert_eq!(mode, ExecutionMode::Parallel);
}

#[test]
fn adaptive_strategy_sparse_no_parallelism() {
    let s = AdaptiveStrategy::new(0.1, 1 << 20, 4);
    let sparse = SparseState::new(10).unwrap();
    let state = AdaptiveState::Sparse {
        state: sparse,
        threshold: 0.1,
    };
    assert!(!s.should_parallelize_gate(&state));
}

// ============================================================================
// GateMatrixCache
// ============================================================================

#[test]
fn cache_insert_and_get() {
    let cache = GateMatrixCache::new(10);
    let key = GateCacheKey {
        gate_name: "H".to_string(),
        params: vec![],
    };
    let isqrt2 = 1.0 / 2.0_f64.sqrt();
    let matrix = [
        [c(isqrt2, 0.0), c(isqrt2, 0.0)],
        [c(isqrt2, 0.0), c(-isqrt2, 0.0)],
    ];
    cache.insert(key.clone(), CachedMatrix::Single(matrix));

    let got = cache.get(&key);
    assert!(got.is_some());
}

#[test]
fn cache_miss_returns_none() {
    let cache = GateMatrixCache::new(10);
    let key = GateCacheKey {
        gate_name: "X".to_string(),
        params: vec![],
    };
    assert!(cache.get(&key).is_none());
}

#[test]
fn cache_hit_rate_tracking() {
    let cache = GateMatrixCache::new(10);
    let key = GateCacheKey {
        gate_name: "X".to_string(),
        params: vec![],
    };
    let matrix = [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
    cache.insert(key.clone(), CachedMatrix::Single(matrix));

    cache.get(&key);
    cache.get(&key);
    let missing_key = GateCacheKey {
        gate_name: "Y".to_string(),
        params: vec![],
    };
    cache.get(&missing_key);

    let rate = cache.hit_rate();
    assert!((rate - 2.0 / 3.0).abs() < 1e-10);
}

#[test]
fn cache_eviction_on_overflow() {
    let cache = GateMatrixCache::new(2);
    let matrix = [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]];

    for i in 0..3 {
        let key = GateCacheKey {
            gate_name: format!("G{}", i),
            params: vec![],
        };
        cache.insert(key, CachedMatrix::Single(matrix));
    }

    // Cache max_size is 2, so one entry was evicted
    let key2 = GateCacheKey {
        gate_name: "G2".to_string(),
        params: vec![],
    };
    assert!(cache.get(&key2).is_some());
}

#[test]
fn cache_clear() {
    let cache = GateMatrixCache::new(10);
    let key = GateCacheKey {
        gate_name: "X".to_string(),
        params: vec![],
    };
    let matrix = [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]];
    cache.insert(key.clone(), CachedMatrix::Single(matrix));
    cache.clear();
    assert!(cache.get(&key).is_none());
}

#[test]
fn ordered_float_equality() {
    let a = OrderedFloat(1.5);
    let b = OrderedFloat(1.5);
    let c = OrderedFloat(2.5);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn cache_key_with_params() {
    let key1 = GateCacheKey {
        gate_name: "RY".to_string(),
        params: vec![OrderedFloat(0.5)],
    };
    let key2 = GateCacheKey {
        gate_name: "RY".to_string(),
        params: vec![OrderedFloat(0.5)],
    };
    let key3 = GateCacheKey {
        gate_name: "RY".to_string(),
        params: vec![OrderedFloat(1.0)],
    };
    assert_eq!(key1, key2);
    assert_ne!(key1, key3);
}

// ============================================================================
// CheckpointManager
// ============================================================================

#[test]
fn checkpoint_create_and_latest() {
    let mut mgr = CheckpointManager::new(5);
    let state = AdaptiveState::new(2).unwrap();
    mgr.create_checkpoint(0, &state).unwrap();
    mgr.create_checkpoint(10, &state).unwrap();

    let latest = mgr.latest_checkpoint().unwrap();
    assert_eq!(latest.gate_index, 10);
}

#[test]
fn checkpoint_max_limit() {
    let mut mgr = CheckpointManager::new(2);
    let state = AdaptiveState::new(1).unwrap();
    for i in 0..5 {
        mgr.create_checkpoint(i, &state).unwrap();
    }
    let latest = mgr.latest_checkpoint().unwrap();
    assert_eq!(latest.gate_index, 4);
}

#[test]
fn checkpoint_clear() {
    let mut mgr = CheckpointManager::new(5);
    let state = AdaptiveState::new(1).unwrap();
    mgr.create_checkpoint(0, &state).unwrap();
    mgr.clear();
    assert!(mgr.latest_checkpoint().is_none());
}

#[test]
fn checkpoint_restore_not_implemented() {
    let mgr = CheckpointManager::new(5);
    assert!(mgr.restore_checkpoint(0).is_err());
}

#[test]
fn checkpoint_with_directory() {
    let mgr = CheckpointManager::new(5).with_directory("/tmp/checkpoints".into());
    assert!(mgr.latest_checkpoint().is_none());
}

// ============================================================================
// ExecutionConfig
// ============================================================================

#[test]
fn execution_config_defaults() {
    let config = ExecutionConfig::default();
    assert_eq!(config.mode, ExecutionMode::Adaptive);
    assert_eq!(config.parallel_strategy, ParallelStrategy::LayerBased);
    assert!(config.use_simd);
    assert!(!config.use_gpu);
    assert!(config.adaptive_state);
    assert!(config.enable_gate_fusion);
    assert!(config.validate().is_ok());
}

#[test]
fn execution_config_performance_preset() {
    let config = ExecutionConfig::performance();
    assert_eq!(config.mode, ExecutionMode::Parallel);
    assert!(config.use_gpu);
    assert!(!config.validate_state);
    assert!(!config.enable_checkpoints);
}

#[test]
fn execution_config_reliable_preset() {
    let config = ExecutionConfig::reliable();
    assert_eq!(config.mode, ExecutionMode::Sequential);
    assert!(config.validate_state);
    assert!(config.enable_checkpoints);
    assert_eq!(config.max_retry_attempts, 5);
}

#[test]
fn execution_config_debug_preset() {
    let config = ExecutionConfig::debug();
    assert_eq!(config.mode, ExecutionMode::Sequential);
    assert!(config.validate_state);
    assert!(!config.enable_gate_fusion);
}

#[test]
fn execution_config_builder_chain() {
    let config = ExecutionConfig::new()
        .with_mode(ExecutionMode::Parallel)
        .with_parallel_threshold(512)
        .with_gpu(true)
        .with_validation(true)
        .with_checkpoints(true, 50)
        .with_timeout(Duration::from_secs(60));

    assert_eq!(config.mode, ExecutionMode::Parallel);
    assert_eq!(config.parallel_threshold, 512);
    assert!(config.use_gpu);
    assert!(config.validate_state);
    assert!(config.enable_checkpoints);
    assert_eq!(config.checkpoint_interval, 50);
    assert_eq!(config.timeout, Some(Duration::from_secs(60)));
}

#[test]
fn execution_config_validation_errors() {
    let config = ExecutionConfig {
        parallel_threshold: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = ExecutionConfig {
        dense_threshold: -0.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = ExecutionConfig {
        dense_threshold: 1.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = ExecutionConfig {
        enable_checkpoints: true,
        checkpoint_interval: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = ExecutionConfig {
        enable_gate_fusion: true,
        max_fusion_size: 0,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let config = ExecutionConfig {
        max_fusion_size: 17,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

// ============================================================================
// RecoveryPolicy
// ============================================================================

#[test]
fn recovery_halt_never_retries() {
    let policy = RecoveryPolicy::Halt;
    assert!(!policy.should_retry(0));
    assert!(!policy.should_retry(1));
    assert_eq!(policy.max_attempts(), 1);
}

#[test]
fn recovery_skip_never_retries() {
    let policy = RecoveryPolicy::Skip;
    assert!(!policy.should_retry(0));
    assert_eq!(policy.max_attempts(), 1);
}

#[test]
fn recovery_retry_once() {
    let policy = RecoveryPolicy::RetryOnce;
    assert!(policy.should_retry(0));
    assert!(policy.should_retry(1));
    assert!(!policy.should_retry(2));
    assert_eq!(policy.max_attempts(), 2);
}

#[test]
fn recovery_retry_with_backoff() {
    let policy = RecoveryPolicy::RetryWithBackoff { max_attempts: 5 };
    assert!(policy.should_retry(0));
    assert!(policy.should_retry(4));
    assert!(!policy.should_retry(5));
    assert_eq!(policy.max_attempts(), 5);
}

#[test]
fn recovery_fallback() {
    let policy = RecoveryPolicy::Fallback;
    assert!(policy.should_retry(0));
    assert!(policy.should_retry(1));
    assert!(!policy.should_retry(2));
    assert_eq!(policy.max_attempts(), 2);
}

#[test]
fn recovery_default_is_halt() {
    let policy = RecoveryPolicy::default();
    assert_eq!(policy, RecoveryPolicy::Halt);
}

// ============================================================================
// ExecutionTelemetry
// ============================================================================

#[test]
fn telemetry_new_is_empty() {
    let t = ExecutionTelemetry::new();
    assert_eq!(t.cache_hits, 0);
    assert_eq!(t.cache_misses, 0);
    assert!(t.error_events.is_empty());
    assert!(t.gate_type_counts.is_empty());
}

#[test]
fn telemetry_log_error() {
    let mut t = ExecutionTelemetry::new();
    t.log_error("something failed");
    assert_eq!(t.error_events.len(), 1);
    assert_eq!(t.error_events[0], "something failed");
}

#[test]
fn telemetry_log_event() {
    let mut t = ExecutionTelemetry::new();
    t.log_event("start_execution");
    assert_eq!(t.custom_events.len(), 1);
    assert_eq!(t.custom_events[0].0, "start_execution");
}

#[test]
fn telemetry_gate_type_counting() {
    let mut t = ExecutionTelemetry::new();
    t.inc_gate_type("H");
    t.inc_gate_type("H");
    t.inc_gate_type("CNOT");
    assert_eq!(t.gate_type_counts["H"], 2);
    assert_eq!(t.gate_type_counts["CNOT"], 1);
}

#[test]
fn telemetry_cache_tracking() {
    let mut t = ExecutionTelemetry::new();
    t.cache_hit();
    t.cache_hit();
    t.cache_miss();
    assert_eq!(t.cache_hits, 2);
    assert_eq!(t.cache_misses, 1);
}

#[test]
fn telemetry_record_memory() {
    let mut t = ExecutionTelemetry::new();
    t.record_memory(1024);
    t.record_memory(2048);
    assert_eq!(t.memory_usage, vec![1024, 2048]);
}

#[test]
fn execution_metrics_from_telemetry() {
    let mut t = ExecutionTelemetry::new();
    t.per_gate_times.push(Duration::from_millis(10));
    t.per_gate_times.push(Duration::from_millis(20));
    t.total_gate_time = Duration::from_millis(30);
    t.cache_hits = 3;
    t.cache_misses = 1;
    t.log_error("err");

    let metrics = ExecutionMetrics::from_telemetry(&t);
    assert_eq!(metrics.gates_executed, 2);
    assert_eq!(metrics.gates_failed, 1);
    assert!((metrics.cache_hit_rate - 0.75).abs() < 1e-10);
    assert_eq!(metrics.total_time, Duration::from_millis(30));
}

#[test]
fn execution_metrics_empty_telemetry() {
    let t = ExecutionTelemetry::new();
    let metrics = ExecutionMetrics::from_telemetry(&t);
    assert_eq!(metrics.gates_executed, 0);
    assert_eq!(metrics.gates_failed, 0);
    assert!((metrics.cache_hit_rate - 0.0).abs() < 1e-10);
    assert_eq!(metrics.average_gate_time, Duration::ZERO);
}

// ============================================================================
// Validation
// ============================================================================

#[test]
fn validation_normalized_state() {
    let state = DenseState::new(1).unwrap();
    let adaptive = AdaptiveState::Dense(state);
    assert!(validate_normalization(&adaptive, 1e-6).is_ok());
    assert!(validate_finite(&adaptive).is_ok());
    assert!(validate_state(&adaptive).is_ok());
}

#[test]
fn validation_unnormalized_state_fails() {
    let amps = vec![c(0.5, 0.0), c(0.5, 0.0)];
    let state = DenseState::from_amplitudes(1, &amps).unwrap();
    let adaptive = AdaptiveState::Dense(state);
    assert!(validate_normalization(&adaptive, 1e-6).is_err());
}

#[test]
fn validation_nan_state_fails() {
    let amps = vec![c(f64::NAN, 0.0), c(0.0, 0.0)];
    let state = DenseState::from_amplitudes(1, &amps).unwrap();
    let adaptive = AdaptiveState::Dense(state);
    assert!(validate_finite(&adaptive).is_err());
}

#[test]
fn validation_inf_state_fails() {
    let amps = vec![c(f64::INFINITY, 0.0), c(0.0, 0.0)];
    let state = DenseState::from_amplitudes(1, &amps).unwrap();
    let adaptive = AdaptiveState::Dense(state);
    assert!(validate_finite(&adaptive).is_err());
}

#[test]
fn validation_sparse_state() {
    let state = SparseState::new(2).unwrap();
    let adaptive = AdaptiveState::Sparse {
        state,
        threshold: 0.1,
    };
    assert!(validate_state(&adaptive).is_ok());
}

// ============================================================================
// ExecutionEngine integration
// ============================================================================

#[test]
fn execution_engine_runs_simple_circuit() {
    let config = ExecutionConfig::new().with_mode(ExecutionMode::Sequential);
    let mut engine = ExecutionEngine::new(config);
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let mut state = AdaptiveState::new(1).unwrap();
    engine.execute(&circuit, &mut state).unwrap();
}

#[test]
fn execution_engine_telemetry_access() {
    let config = ExecutionConfig::new().with_mode(ExecutionMode::Sequential);
    let mut engine = ExecutionEngine::new(config);
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    let mut state = AdaptiveState::new(1).unwrap();
    engine.execute(&circuit, &mut state).unwrap();

    let telemetry = engine.get_telemetry();
    assert!(!telemetry.per_gate_times.is_empty());
}

#[test]
fn execution_engine_multi_qubit_circuit() {
    let config = ExecutionConfig::new().with_mode(ExecutionMode::Sequential);
    let mut engine = ExecutionEngine::new(config);
    let mut circuit = Circuit::new(3);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    circuit.add_gate(Arc::new(CNot), &[q(1), q(2)]).unwrap();
    let mut state = AdaptiveState::new(3).unwrap();
    engine.execute(&circuit, &mut state).unwrap();
}
