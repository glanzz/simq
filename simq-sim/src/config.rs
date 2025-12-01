//! Simulator configuration

/// Configuration for the quantum simulator
#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Enable GPU backend (wgpu)
    /// When true, uses GPU for gate application if available.
    /// Default: false
    pub use_gpu: bool,
    /// Density threshold for switching from sparse to dense representation
    ///
    /// When the state density (fraction of non-zero amplitudes) exceeds this
    /// threshold, the simulator automatically switches to dense representation.
    ///
    /// Default: 0.1 (10%)
    pub sparse_threshold: f64,

    /// Minimum number of qubits to enable parallel execution
    ///
    /// Circuits with fewer qubits use single-threaded execution to avoid
    /// synchronization overhead.
    ///
    /// Default: 8
    pub parallel_threshold: usize,

    /// Number of measurement shots for sampling
    ///
    /// When performing final measurements, this determines how many samples
    /// to draw from the probability distribution.
    ///
    /// Default: 1024
    pub shots: usize,

    /// Enable circuit compilation and optimization
    ///
    /// When true, circuits are optimized before execution using the compiler.
    ///
    /// Default: true
    pub optimize_circuit: bool,

    /// Optimization level (0-3)
    ///
    /// - O0: No optimization
    /// - O1: Basic optimizations (dead code elimination)
    /// - O2: Standard optimizations (fusion, commutation)
    /// - O3: Aggressive optimizations (all passes)
    ///
    /// Default: 2 (O2)
    pub optimization_level: u8,

    /// Enable execution statistics collection
    ///
    /// When true, collects detailed timing and resource usage statistics.
    ///
    /// Default: false
    pub collect_statistics: bool,

    /// Random number generator seed for reproducibility
    ///
    /// If None, uses a random seed. Set to Some(seed) for deterministic results.
    ///
    /// Default: None (random)
    pub seed: Option<u64>,

    /// Memory limit in bytes
    ///
    /// Maximum memory to use for state vectors. If exceeded, returns an error.
    /// Set to 0 for no limit.
    ///
    /// Default: 0 (unlimited)
    pub memory_limit: usize,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            sparse_threshold: 0.1,
            parallel_threshold: 8,
            shots: 1024,
            optimize_circuit: true,
            optimization_level: 2,
            collect_statistics: false,
            seed: None,
            memory_limit: 0,
            use_gpu: false,
        }
    }
}

impl SimulatorConfig {
    /// Create a new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for speed
    ///
    /// - Aggressive optimization (O3)
    /// - No statistics collection
    /// - Lower parallel threshold
    pub fn fast() -> Self {
        Self {
            optimization_level: 3,
            collect_statistics: false,
            parallel_threshold: 6,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for accuracy
    ///
    /// - No optimization (to preserve exact circuit)
    /// - More measurement shots
    /// - Statistics collection enabled
    pub fn accurate() -> Self {
        Self {
            optimize_circuit: false,
            optimization_level: 0,
            shots: 10000,
            collect_statistics: true,
            ..Default::default()
        }
    }

    /// Create a configuration for debugging
    ///
    /// - No optimization
    /// - Statistics collection
    /// - Deterministic seed
    pub fn debug() -> Self {
        Self {
            optimize_circuit: false,
            optimization_level: 0,
            collect_statistics: true,
            seed: Some(42),
            ..Default::default()
        }
    }

    /// Set the sparse threshold
    pub fn with_sparse_threshold(mut self, threshold: f64) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Set the number of measurement shots
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Set the optimization level
    pub fn with_optimization_level(mut self, level: u8) -> Self {
        self.optimization_level = level.min(3);
        self
    }

    /// Enable or disable circuit optimization
    pub fn with_optimization(mut self, enabled: bool) -> Self {
        self.optimize_circuit = enabled;
        self
    }

    /// Set the random seed for deterministic execution
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable statistics collection
    pub fn with_statistics(mut self, enabled: bool) -> Self {
        self.collect_statistics = enabled;
        self
    }

    /// Set memory limit in bytes
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.sparse_threshold < 0.0 || self.sparse_threshold > 1.0 {
            return Err(format!(
                "sparse_threshold must be in [0,1], got {}",
                self.sparse_threshold
            ));
        }

        if self.shots == 0 {
            return Err("shots must be > 0".to_string());
        }

        if self.optimization_level > 3 {
            return Err(format!("optimization_level must be 0-3, got {}", self.optimization_level));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SimulatorConfig::default();
        assert_eq!(config.sparse_threshold, 0.1);
        assert_eq!(config.parallel_threshold, 8);
        assert_eq!(config.shots, 1024);
        assert!(config.optimize_circuit);
        assert_eq!(config.optimization_level, 2);
    }

    #[test]
    fn test_fast_config() {
        let config = SimulatorConfig::fast();
        assert_eq!(config.optimization_level, 3);
        assert!(!config.collect_statistics);
        assert_eq!(config.parallel_threshold, 6);
    }

    #[test]
    fn test_accurate_config() {
        let config = SimulatorConfig::accurate();
        assert!(!config.optimize_circuit);
        assert_eq!(config.shots, 10000);
        assert!(config.collect_statistics);
    }

    #[test]
    fn test_builder_pattern() {
        let config = SimulatorConfig::new()
            .with_shots(2048)
            .with_optimization_level(3)
            .with_seed(42);

        assert_eq!(config.shots, 2048);
        assert_eq!(config.optimization_level, 3);
        assert_eq!(config.seed, Some(42));
    }

    #[test]
    fn test_validate() {
        let config = SimulatorConfig::default();
        assert!(config.validate().is_ok());

        let invalid = SimulatorConfig {
            sparse_threshold: 1.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid = SimulatorConfig {
            shots: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
