//! Telemetry and metrics collection

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Telemetry data for execution profiling
#[derive(Debug, Default, Clone)]
pub struct ExecutionTelemetry {
    pub total_gate_time: Duration,
    pub per_gate_times: Vec<Duration>,
    pub state_density: Vec<f32>,
    pub memory_usage: Vec<usize>,
    pub gate_type_counts: HashMap<String, usize>,
    pub thread_ids: Vec<u64>,
    pub parallelism: usize,
    pub error_events: Vec<String>,
    pub custom_events: Vec<(String, Instant)>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub sparse_dense_transitions: usize,
}

impl ExecutionTelemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn log_error(&mut self, msg: impl Into<String>) {
        self.error_events.push(msg.into());
    }

    pub fn log_event(&mut self, label: impl Into<String>) {
        self.custom_events.push((label.into(), Instant::now()));
    }

    pub fn inc_gate_type(&mut self, gate_name: &str) {
        *self
            .gate_type_counts
            .entry(gate_name.to_string())
            .or_insert(0) += 1;
    }

    pub fn record_memory(&mut self, bytes: usize) {
        self.memory_usage.push(bytes);
    }

    pub fn record_thread(&mut self) {
        self.thread_ids.push(thread_id::get() as u64);
    }

    pub fn cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn cache_miss(&mut self) {
        self.cache_misses += 1;
    }
}

/// Real-time execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    pub gates_executed: usize,
    pub gates_failed: usize,
    pub average_gate_time: Duration,
    pub total_time: Duration,
    pub cache_hit_rate: f64,
}

impl ExecutionMetrics {
    pub fn from_telemetry(telemetry: &ExecutionTelemetry) -> Self {
        let total_cache = telemetry.cache_hits + telemetry.cache_misses;
        let cache_hit_rate = if total_cache > 0 {
            telemetry.cache_hits as f64 / total_cache as f64
        } else {
            0.0
        };

        let avg_time = if !telemetry.per_gate_times.is_empty() {
            telemetry.total_gate_time / telemetry.per_gate_times.len() as u32
        } else {
            Duration::ZERO
        };

        Self {
            gates_executed: telemetry.per_gate_times.len(),
            gates_failed: telemetry.error_events.len(),
            average_gate_time: avg_time,
            total_time: telemetry.total_gate_time,
            cache_hit_rate,
        }
    }
}
