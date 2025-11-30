//! Execution engine configuration

use std::time::Duration;

/// Configuration for the execution engine
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Execution mode (sequential, parallel, adaptive)
    pub mode: ExecutionMode,

    /// Parallel execution strategy
    pub parallel_strategy: ParallelStrategy,

    /// Threshold for using parallel execution (state vector size)
    pub parallel_threshold: usize,

    /// Enable SIMD optimizations
    pub use_simd: bool,

    /// Enable GPU acceleration
    pub use_gpu: bool,

    /// GPU device index (if multiple GPUs available)
    pub gpu_device_id: usize,

    /// Enable automatic sparse/dense adaptation
    pub adaptive_state: bool,

    /// Density threshold for sparse->dense conversion
    pub dense_threshold: f32,

    /// Enable state validation after each gate
    pub validate_state: bool,

    /// Enable checkpointing
    pub enable_checkpoints: bool,

    /// Checkpoint interval (number of gates)
    pub checkpoint_interval: usize,

    /// Enable gate fusion optimization
    pub enable_gate_fusion: bool,

    /// Maximum number of gates to fuse
    pub max_fusion_size: usize,

    /// Execution timeout (None = no timeout)
    pub timeout: Option<Duration>,

    /// Maximum retry attempts for failed gates
    pub max_retry_attempts: usize,

    /// Enable telemetry collection
    pub collect_telemetry: bool,

    /// Cache size for gate matrices (number of gates)
    pub matrix_cache_size: usize,

    /// Number of worker threads (0 = use all cores)
    pub num_threads: usize,

    /// Batch size for parallel gate execution
    pub batch_size: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Adaptive,
            parallel_strategy: ParallelStrategy::LayerBased,
            parallel_threshold: 1 << 10, // 1024 amplitudes
            use_simd: true,
            use_gpu: false,
            gpu_device_id: 0,
            adaptive_state: true,
            dense_threshold: 0.1, // 10%
            validate_state: cfg!(debug_assertions),
            enable_checkpoints: false,
            checkpoint_interval: 100,
            enable_gate_fusion: true,
            max_fusion_size: 4,
            timeout: None,
            max_retry_attempts: 2,
            collect_telemetry: true,
            matrix_cache_size: 128,
            num_threads: 0, // Auto-detect
            batch_size: 64,
        }
    }
}

impl ExecutionConfig {
    /// Create a new execution config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure for maximum performance
    pub fn performance() -> Self {
        Self {
            mode: ExecutionMode::Parallel,
            parallel_strategy: ParallelStrategy::LayerBased,
            parallel_threshold: 1 << 8,
            use_simd: true,
            use_gpu: true,
            adaptive_state: true,
            validate_state: false,
            enable_checkpoints: false,
            enable_gate_fusion: true,
            max_fusion_size: 8,
            collect_telemetry: false,
            ..Default::default()
        }
    }

    /// Configure for maximum reliability
    pub fn reliable() -> Self {
        Self {
            mode: ExecutionMode::Sequential,
            validate_state: true,
            enable_checkpoints: true,
            checkpoint_interval: 50,
            max_retry_attempts: 5,
            collect_telemetry: true,
            ..Default::default()
        }
    }

    /// Configure for debugging
    pub fn debug() -> Self {
        Self {
            mode: ExecutionMode::Sequential,
            validate_state: true,
            enable_checkpoints: true,
            checkpoint_interval: 10,
            collect_telemetry: true,
            enable_gate_fusion: false,
            ..Default::default()
        }
    }

    /// Builder: set execution mode
    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder: set parallel threshold
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Builder: enable/disable GPU
    pub fn with_gpu(mut self, enable: bool) -> Self {
        self.use_gpu = enable;
        self
    }

    /// Builder: set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Builder: enable/disable validation
    pub fn with_validation(mut self, enable: bool) -> Self {
        self.validate_state = enable;
        self
    }

    /// Builder: enable/disable checkpoints
    pub fn with_checkpoints(mut self, enable: bool, interval: usize) -> Self {
        self.enable_checkpoints = enable;
        self.checkpoint_interval = interval;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.parallel_threshold == 0 {
            return Err("parallel_threshold must be > 0".to_string());
        }

        if self.dense_threshold < 0.0 || self.dense_threshold > 1.0 {
            return Err("dense_threshold must be between 0.0 and 1.0".to_string());
        }

        if self.checkpoint_interval == 0 && self.enable_checkpoints {
            return Err("checkpoint_interval must be > 0 when checkpoints enabled".to_string());
        }

        if self.max_fusion_size == 0 && self.enable_gate_fusion {
            return Err("max_fusion_size must be > 0 when gate fusion enabled".to_string());
        }

        if self.max_fusion_size > 16 {
            return Err("max_fusion_size too large (max: 16)".to_string());
        }

        Ok(())
    }
}

/// Execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Execute gates sequentially (safest, slowest)
    Sequential,

    /// Execute gates in parallel when possible
    Parallel,

    /// Adaptively choose based on circuit structure and state
    Adaptive,

    /// GPU-accelerated execution
    Gpu,
}

/// Parallel execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// Execute independent gates in layers (default)
    LayerBased,

    /// Parallelize within gate application (data parallelism)
    DataParallel,

    /// Hybrid: combine layer and data parallelism
    Hybrid,

    /// Task-based parallelism with work stealing
    TaskBased,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ExecutionConfig::default();
        assert_eq!(config.mode, ExecutionMode::Adaptive);
        assert!(config.use_simd);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_performance_config() {
        let config = ExecutionConfig::performance();
        assert_eq!(config.mode, ExecutionMode::Parallel);
        assert!(config.use_gpu);
        assert!(!config.validate_state);
    }

    #[test]
    fn test_reliable_config() {
        let config = ExecutionConfig::reliable();
        assert_eq!(config.mode, ExecutionMode::Sequential);
        assert!(config.validate_state);
        assert!(config.enable_checkpoints);
    }

    #[test]
    fn test_validation() {
        let config = ExecutionConfig {
            parallel_threshold: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = ExecutionConfig {
            parallel_threshold: 1024,
            dense_threshold: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = ExecutionConfig {
            parallel_threshold: 1024,
            dense_threshold: 0.1,
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = ExecutionConfig::new()
            .with_mode(ExecutionMode::Parallel)
            .with_gpu(true)
            .with_validation(false);

        assert_eq!(config.mode, ExecutionMode::Parallel);
        assert!(config.use_gpu);
        assert!(!config.validate_state);
    }
}
