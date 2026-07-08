//! Lazy gate evaluation for optimized circuit execution
//!
//! This module implements lazy evaluation of quantum gates, deferring matrix
//! computation and gate application until absolutely necessary. This provides
//! several benefits:
//!
//! 1. **Deferred Computation**: Gate matrices are only computed when needed
//! 2. **Matrix Caching**: Computed matrices are cached for reuse (especially useful for parameterized circuits)
//! 3. **Fusion Opportunities**: Gates can be fused before computing their matrices
//! 4. **Batch Processing**: Operations can be batched for better cache locality
//!
//! # Theory
//!
//! Traditional eager evaluation computes gate matrices immediately:
//! ```text
//! Define Gate → Compute Matrix → Store in Circuit → Execute
//! ```
//!
//! Lazy evaluation defers computation:
//! ```text
//! Define Gate → Store Reference → [Optimize/Fuse] → Compute on Demand → Execute → Cache
//! ```
//!
//! This is particularly beneficial for:
//! - **Parameterized circuits** (VQE, QAOA): Gates defined once, executed many times with different parameters
//! - **Large circuits**: Matrices computed only for gates that actually execute
//! - **Gate fusion**: Avoid computing matrices for gates that will be fused
//!
//! # Example
//!
//! ```ignore
//! use ferriq_compiler::lazy::{LazyExecutor, LazyConfig};
//! use ferriq_core::Circuit;
//! use ferriq_state::StateVector;
//!
//! let circuit = /* build circuit */;
//! let mut state = StateVector::new(num_qubits);
//!
//! // Create lazy executor
//! let mut executor = LazyExecutor::new(LazyConfig::default());
//!
//! // Execute with lazy evaluation
//! executor.execute(&circuit, &mut state)?;
//!
//! // Matrix cache is now populated for reuse
//! executor.execute(&circuit, &mut state)?; // Faster on second run
//! ```

use crate::fusion::FusionConfig;
use ahash::AHashMap;
use num_complex::Complex64;
use ferriq_core::{gate::Gate, Circuit, GateOp, QuantumError, Result};
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Configuration for lazy evaluation
#[derive(Debug, Clone)]
pub struct LazyConfig {
    /// Enable matrix caching (default: true)
    /// Caches computed matrices for reuse across executions
    pub enable_caching: bool,

    /// Maximum cache size in entries (default: 1000)
    /// LRU eviction when exceeded
    pub max_cache_size: usize,

    /// Enable automatic fusion of adjacent single-qubit gates (default: true)
    /// Fuse gates before computing matrices
    pub enable_fusion: bool,

    /// Fusion configuration
    pub fusion_config: FusionConfig,

    /// Batch size for gate application (default: 16)
    /// Gates are batched for better cache locality
    pub batch_size: usize,
}

impl Default for LazyConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            max_cache_size: 1000,
            enable_fusion: true,
            fusion_config: FusionConfig::default(),
            batch_size: 16,
        }
    }
}

/// A wrapper around a gate that defers matrix computation
///
/// The matrix is computed on first access and optionally cached.
pub struct LazyGate {
    /// The underlying gate
    gate: Arc<dyn Gate>,

    /// Lazily computed matrix (for single-qubit gates)
    matrix_1q: RefCell<Option<[[Complex64; 2]; 2]>>,

    /// Lazily computed matrix (for two-qubit gates)
    matrix_2q: RefCell<Option<[[Complex64; 4]; 4]>>,
}

impl LazyGate {
    /// Create a new lazy gate wrapper
    pub fn new(gate: Arc<dyn Gate>) -> Self {
        Self {
            gate,
            matrix_1q: RefCell::new(None),
            matrix_2q: RefCell::new(None),
        }
    }

    /// Get the underlying gate
    pub fn gate(&self) -> &Arc<dyn Gate> {
        &self.gate
    }

    /// Get the gate matrix for a single-qubit gate, computing it if necessary
    pub fn matrix_1q(&self) -> Result<[[Complex64; 2]; 2]> {
        if self.gate.num_qubits() != 1 {
            return Err(QuantumError::ValidationError("Expected single-qubit gate".to_string()));
        }

        // Check if already computed
        if let Some(matrix) = *self.matrix_1q.borrow() {
            return Ok(matrix);
        }

        // Compute matrix
        let matrix_vec = self.gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError(format!(
                "Gate {} does not provide a matrix",
                self.gate.name()
            ))
        })?;

        if matrix_vec.len() != 4 {
            return Err(QuantumError::ValidationError(format!(
                "Expected 2x2 matrix (4 elements), got {}",
                matrix_vec.len()
            )));
        }

        // Convert to 2D array
        let matrix = [
            [matrix_vec[0], matrix_vec[1]],
            [matrix_vec[2], matrix_vec[3]],
        ];

        // Cache it
        *self.matrix_1q.borrow_mut() = Some(matrix);

        Ok(matrix)
    }

    /// Get the gate matrix for a two-qubit gate, computing it if necessary
    pub fn matrix_2q(&self) -> Result<[[Complex64; 4]; 4]> {
        if self.gate.num_qubits() != 2 {
            return Err(QuantumError::ValidationError("Expected two-qubit gate".to_string()));
        }

        // Check if already computed
        if let Some(matrix) = *self.matrix_2q.borrow() {
            return Ok(matrix);
        }

        // Compute matrix
        let matrix_vec = self.gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError(format!(
                "Gate {} does not provide a matrix",
                self.gate.name()
            ))
        })?;

        if matrix_vec.len() != 16 {
            return Err(QuantumError::ValidationError(format!(
                "Expected 4x4 matrix (16 elements), got {}",
                matrix_vec.len()
            )));
        }

        // Convert to 2D array
        let matrix = [
            [matrix_vec[0], matrix_vec[1], matrix_vec[2], matrix_vec[3]],
            [matrix_vec[4], matrix_vec[5], matrix_vec[6], matrix_vec[7]],
            [matrix_vec[8], matrix_vec[9], matrix_vec[10], matrix_vec[11]],
            [
                matrix_vec[12],
                matrix_vec[13],
                matrix_vec[14],
                matrix_vec[15],
            ],
        ];

        // Cache it
        *self.matrix_2q.borrow_mut() = Some(matrix);

        Ok(matrix)
    }
}

impl std::fmt::Debug for LazyGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyGate")
            .field("gate", &self.gate.name())
            .field("matrix_1q_computed", &self.matrix_1q.borrow().is_some())
            .field("matrix_2q_computed", &self.matrix_2q.borrow().is_some())
            .finish()
    }
}

/// Key for caching gate matrices
///
/// Uses gate name and a hash of the matrix (for parameterized gates)
#[derive(Clone, Eq, PartialEq, Hash)]
struct MatrixCacheKey {
    gate_name: String,
    num_qubits: usize,
    // For parameterized gates, we need to distinguish different parameter values
    // We use a hash of the matrix as a proxy
    matrix_hash: u64,
}

impl MatrixCacheKey {
    fn from_gate(gate: &Arc<dyn Gate>) -> Self {
        let gate_name = gate.name().to_string();
        let num_qubits = gate.num_qubits();

        // Compute hash of matrix if available
        let matrix_hash = if let Some(matrix) = gate.matrix() {
            let mut hasher = ahash::AHasher::default();
            for elem in &matrix {
                // Hash real and imaginary parts
                hasher.write_u64(elem.re.to_bits());
                hasher.write_u64(elem.im.to_bits());
            }
            hasher.finish()
        } else {
            0
        };

        Self {
            gate_name,
            num_qubits,
            matrix_hash,
        }
    }
}

/// Cache for computed gate matrices
pub struct MatrixCache {
    /// Single-qubit gate matrices
    cache_1q: AHashMap<MatrixCacheKey, [[Complex64; 2]; 2]>,

    /// Two-qubit gate matrices
    cache_2q: AHashMap<MatrixCacheKey, [[Complex64; 4]; 4]>,

    /// Maximum cache size
    max_size: usize,

    /// Access order for LRU eviction (simple counter)
    access_order: AHashMap<MatrixCacheKey, u64>,
    access_counter: u64,
}

impl MatrixCache {
    /// Create a new matrix cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache_1q: AHashMap::with_capacity(max_size / 2),
            cache_2q: AHashMap::with_capacity(max_size / 2),
            max_size,
            access_order: AHashMap::with_capacity(max_size),
            access_counter: 0,
        }
    }

    /// Get or compute a single-qubit gate matrix
    pub fn get_or_compute_1q(&mut self, gate: &Arc<dyn Gate>) -> Result<[[Complex64; 2]; 2]> {
        let key = MatrixCacheKey::from_gate(gate);

        // Check cache
        if let Some(matrix) = self.cache_1q.get(&key) {
            // Update access order
            self.access_counter += 1;
            self.access_order.insert(key.clone(), self.access_counter);
            return Ok(*matrix);
        }

        // Compute matrix
        let matrix_vec = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError(format!("Gate {} does not provide a matrix", gate.name()))
        })?;

        if matrix_vec.len() != 4 {
            return Err(QuantumError::ValidationError(format!(
                "Expected 2x2 matrix (4 elements), got {}",
                matrix_vec.len()
            )));
        }

        let matrix = [
            [matrix_vec[0], matrix_vec[1]],
            [matrix_vec[2], matrix_vec[3]],
        ];

        // Evict if necessary
        if self.cache_1q.len() + self.cache_2q.len() >= self.max_size {
            self.evict_lru();
        }

        // Store in cache
        self.cache_1q.insert(key.clone(), matrix);
        self.access_counter += 1;
        self.access_order.insert(key, self.access_counter);

        Ok(matrix)
    }

    /// Get or compute a two-qubit gate matrix
    pub fn get_or_compute_2q(&mut self, gate: &Arc<dyn Gate>) -> Result<[[Complex64; 4]; 4]> {
        let key = MatrixCacheKey::from_gate(gate);

        // Check cache
        if let Some(matrix) = self.cache_2q.get(&key) {
            // Update access order
            self.access_counter += 1;
            self.access_order.insert(key.clone(), self.access_counter);
            return Ok(*matrix);
        }

        // Compute matrix
        let matrix_vec = gate.matrix().ok_or_else(|| {
            QuantumError::ValidationError(format!("Gate {} does not provide a matrix", gate.name()))
        })?;

        if matrix_vec.len() != 16 {
            return Err(QuantumError::ValidationError(format!(
                "Expected 4x4 matrix (16 elements), got {}",
                matrix_vec.len()
            )));
        }

        let matrix = [
            [matrix_vec[0], matrix_vec[1], matrix_vec[2], matrix_vec[3]],
            [matrix_vec[4], matrix_vec[5], matrix_vec[6], matrix_vec[7]],
            [matrix_vec[8], matrix_vec[9], matrix_vec[10], matrix_vec[11]],
            [
                matrix_vec[12],
                matrix_vec[13],
                matrix_vec[14],
                matrix_vec[15],
            ],
        ];

        // Evict if necessary
        if self.cache_1q.len() + self.cache_2q.len() >= self.max_size {
            self.evict_lru();
        }

        // Store in cache
        self.cache_2q.insert(key.clone(), matrix);
        self.access_counter += 1;
        self.access_order.insert(key, self.access_counter);

        Ok(matrix)
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some((lru_key, _)) = self.access_order.iter().min_by_key(|(_, &count)| count) {
            let lru_key = lru_key.clone();
            self.cache_1q.remove(&lru_key);
            self.cache_2q.remove(&lru_key);
            self.access_order.remove(&lru_key);
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache_1q.clear();
        self.cache_2q.clear();
        self.access_order.clear();
        self.access_counter = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries_1q: self.cache_1q.len(),
            entries_2q: self.cache_2q.len(),
            total_entries: self.cache_1q.len() + self.cache_2q.len(),
            max_size: self.max_size,
        }
    }
}

/// Statistics about the matrix cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries_1q: usize,
    pub entries_2q: usize,
    pub total_entries: usize,
    pub max_size: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache: {}/{} entries (1q: {}, 2q: {})",
            self.total_entries, self.max_size, self.entries_1q, self.entries_2q
        )
    }
}

/// Statistics for lazy executor
#[derive(Debug, Clone, Default)]
pub struct ExecutorStats {
    /// Number of matrix computations
    pub matrices_computed: usize,

    /// Number of cache hits
    pub cache_hits: usize,

    /// Number of cache misses
    pub cache_misses: usize,

    /// Number of gates fused
    pub gates_fused: usize,

    /// Number of batches executed
    pub batches_executed: usize,
}

impl ExecutorStats {
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

impl std::fmt::Display for ExecutorStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Executor Stats:\n  Matrices computed: {}\n  Cache hits: {} ({:.1}%)\n  Cache misses: {}\n  Gates fused: {}\n  Batches: {}",
            self.matrices_computed,
            self.cache_hits,
            self.cache_hit_rate() * 100.0,
            self.cache_misses,
            self.gates_fused,
            self.batches_executed
        )
    }
}

/// Lazy executor for circuit execution with deferred computation
///
/// This executor applies gates to a state vector with lazy evaluation,
/// deferring matrix computation until necessary and optionally caching results.
pub struct LazyExecutor {
    config: LazyConfig,
    cache: MatrixCache,
    stats: ExecutorStats,
}

impl LazyExecutor {
    /// Create a new lazy executor with the given configuration
    pub fn new(config: LazyConfig) -> Self {
        let cache = MatrixCache::new(config.max_cache_size);
        Self {
            config,
            cache,
            stats: ExecutorStats::default(),
        }
    }

    /// Execute a circuit on a state vector with lazy evaluation
    ///
    /// This method applies all gates in the circuit to the state vector,
    /// using lazy evaluation and caching as configured.
    pub fn execute(&mut self, circuit: &Circuit, state: &mut [Complex64]) -> Result<()> {
        let num_qubits = circuit.num_qubits();

        // Validate state size
        let expected_size = 1 << num_qubits;
        if state.len() != expected_size {
            return Err(QuantumError::ValidationError(format!(
                "State size {} does not match circuit with {} qubits (expected {})",
                state.len(),
                num_qubits,
                expected_size
            )));
        }

        // Apply fusion if enabled
        let circuit = if self.config.enable_fusion {
            match crate::fusion::fuse_single_qubit_gates(
                circuit,
                Some(self.config.fusion_config.clone()),
            ) {
                Ok(fused) => {
                    let original_len = circuit.len();
                    let fused_len = fused.len();
                    if fused_len < original_len {
                        self.stats.gates_fused += original_len - fused_len;
                    }
                    fused
                },
                Err(_) => circuit.clone(),
            }
        } else {
            circuit.clone()
        };

        // Execute gates in batches
        let ops: Vec<_> = circuit.operations().collect();
        for batch in ops.chunks(self.config.batch_size) {
            self.execute_batch(batch, state, num_qubits)?;
            self.stats.batches_executed += 1;
        }

        Ok(())
    }

    /// Execute a batch of gates
    fn execute_batch(
        &mut self,
        batch: &[&GateOp],
        state: &mut [Complex64],
        num_qubits: usize,
    ) -> Result<()> {
        for gate_op in batch {
            self.apply_gate(gate_op, state, num_qubits)?;
        }
        Ok(())
    }

    /// Apply a single gate operation to the state
    fn apply_gate(
        &mut self,
        gate_op: &GateOp,
        state: &mut [Complex64],
        num_qubits: usize,
    ) -> Result<()> {
        let gate = gate_op.gate();
        let qubits = gate_op.qubits();

        match gate.num_qubits() {
            1 => {
                // Single-qubit gate
                let matrix = if self.config.enable_caching {
                    let cached = self
                        .cache
                        .cache_1q
                        .contains_key(&MatrixCacheKey::from_gate(gate));
                    let result = self.cache.get_or_compute_1q(gate)?;

                    if cached {
                        self.stats.cache_hits += 1;
                    } else {
                        self.stats.cache_misses += 1;
                        self.stats.matrices_computed += 1;
                    }
                    result
                } else {
                    // No caching - compute directly
                    self.stats.matrices_computed += 1;
                    let lazy = LazyGate::new(gate.clone());
                    lazy.matrix_1q()?
                };

                // Apply using SIMD
                ferriq_state::simd::apply_single_qubit_gate(
                    state,
                    &matrix,
                    qubits[0].index(),
                    num_qubits,
                );
            },
            2 => {
                // Two-qubit gate
                let matrix = if self.config.enable_caching {
                    let cached = self
                        .cache
                        .cache_2q
                        .contains_key(&MatrixCacheKey::from_gate(gate));
                    let result = self.cache.get_or_compute_2q(gate)?;

                    if cached {
                        self.stats.cache_hits += 1;
                    } else {
                        self.stats.cache_misses += 1;
                        self.stats.matrices_computed += 1;
                    }
                    result
                } else {
                    // No caching - compute directly
                    self.stats.matrices_computed += 1;
                    let lazy = LazyGate::new(gate.clone());
                    lazy.matrix_2q()?
                };

                // Apply using SIMD
                ferriq_state::simd::apply_two_qubit_gate(
                    state,
                    &matrix,
                    qubits[0].index(),
                    qubits[1].index(),
                    num_qubits,
                );
            },
            n => {
                return Err(QuantumError::ValidationError(format!(
                    "Lazy executor only supports 1 and 2-qubit gates, got {} qubits",
                    n
                )));
            },
        }

        Ok(())
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutorStats {
        &self.stats
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ExecutorStats::default();
    }

    /// Clear the matrix cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the configuration
    pub fn config(&self) -> &LazyConfig {
        &self.config
    }
}

impl std::fmt::Debug for LazyExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyExecutor")
            .field("config", &self.config)
            .field("cache_stats", &self.cache.stats())
            .field("stats", &self.stats)
            .finish()
    }
}

impl Default for LazyExecutor {
    fn default() -> Self {
        Self::new(LazyConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferriq_core::gate::Gate;
    use ferriq_gates::standard::{CNot, Hadamard, PauliX};

    // -----------------------------------------------------------------------
    // Mock gate helpers for error-path testing
    // -----------------------------------------------------------------------

    /// A mock 2-qubit gate used to test calling matrix_1q() on a 2-qubit gate.
    #[derive(Debug)]
    struct MockTwoQubitGate;

    impl Gate for MockTwoQubitGate {
        fn name(&self) -> &str {
            "Mock2Q"
        }
        fn num_qubits(&self) -> usize {
            2
        }
        fn matrix(&self) -> Option<Vec<Complex64>> {
            // Return a valid 16-element 4x4 matrix (CNOT-like)
            Some(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ])
        }
    }

    /// A mock 1-qubit gate that returns no matrix (matrix() == None).
    #[derive(Debug)]
    struct MockNoMatrixGate;

    impl Gate for MockNoMatrixGate {
        fn name(&self) -> &str {
            "MockNoMatrix"
        }
        fn num_qubits(&self) -> usize {
            1
        }
        fn matrix(&self) -> Option<Vec<Complex64>> {
            None
        }
    }

    /// A mock 1-qubit gate whose matrix() returns wrong number of elements.
    #[derive(Debug)]
    struct MockBadMatrix1qGate;

    impl Gate for MockBadMatrix1qGate {
        fn name(&self) -> &str {
            "MockBadMatrix1q"
        }
        fn num_qubits(&self) -> usize {
            1
        }
        fn matrix(&self) -> Option<Vec<Complex64>> {
            // Return only 2 elements instead of 4
            Some(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
        }
    }

    /// A mock 2-qubit gate that returns no matrix.
    #[derive(Debug)]
    struct MockNoMatrix2qGate;

    impl Gate for MockNoMatrix2qGate {
        fn name(&self) -> &str {
            "MockNoMatrix2q"
        }
        fn num_qubits(&self) -> usize {
            2
        }
        fn matrix(&self) -> Option<Vec<Complex64>> {
            None
        }
    }

    /// A mock 2-qubit gate whose matrix() returns wrong number of elements.
    #[derive(Debug)]
    struct MockBadMatrix2qGate;

    impl Gate for MockBadMatrix2qGate {
        fn name(&self) -> &str {
            "MockBadMatrix2q"
        }
        fn num_qubits(&self) -> usize {
            2
        }
        fn matrix(&self) -> Option<Vec<Complex64>> {
            // Return only 4 elements instead of 16
            Some(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ])
        }
    }

    // -----------------------------------------------------------------------
    // Existing tests (unchanged)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lazy_gate_single_qubit() {
        let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);

        // Matrix should not be computed yet
        assert!(lazy.matrix_1q.borrow().is_none());

        // First access computes matrix
        let matrix1 = lazy.matrix_1q().unwrap();
        assert!(lazy.matrix_1q.borrow().is_some());

        // Second access uses cached matrix
        let matrix2 = lazy.matrix_1q().unwrap();
        assert_eq!(matrix1, matrix2);
    }

    #[test]
    fn test_matrix_cache() {
        let mut cache = MatrixCache::new(10);

        let h_gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let x_gate = Arc::new(PauliX) as Arc<dyn Gate>;

        // First access should compute
        let h_matrix1 = cache.get_or_compute_1q(&h_gate).unwrap();

        // Second access should hit cache
        let h_matrix2 = cache.get_or_compute_1q(&h_gate).unwrap();
        assert_eq!(h_matrix1, h_matrix2);

        // Different gate should compute new matrix
        let x_matrix = cache.get_or_compute_1q(&x_gate).unwrap();
        assert_ne!(h_matrix1, x_matrix);

        let stats = cache.stats();
        assert_eq!(stats.entries_1q, 2);
        assert_eq!(stats.total_entries, 2);
    }

    #[test]
    fn test_matrix_cache_lru_eviction() {
        let mut cache = MatrixCache::new(2);

        let h_gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let x_gate = Arc::new(PauliX) as Arc<dyn Gate>;

        // Fill cache
        cache.get_or_compute_1q(&h_gate).unwrap();
        cache.get_or_compute_1q(&x_gate).unwrap();

        assert_eq!(cache.stats().total_entries, 2);

        // Access H again to make it more recent
        cache.get_or_compute_1q(&h_gate).unwrap();

        // Add another gate - should evict X (least recently used)
        let z_gate = Arc::new(ferriq_gates::standard::PauliZ) as Arc<dyn Gate>;
        cache.get_or_compute_1q(&z_gate).unwrap();

        // Cache should still have 2 entries
        assert_eq!(cache.stats().total_entries, 2);
    }

    // -----------------------------------------------------------------------
    // New tests to cover previously uncovered lines
    // -----------------------------------------------------------------------

    // --- LazyGate::gate() (lines 117-118) ---

    #[test]
    fn test_lazy_gate_gate_accessor() {
        let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate.clone());
        // gate() should return a reference to the stored gate
        assert_eq!(lazy.gate().name(), "H");
    }

    // --- LazyGate::matrix_1q() error: wrong num_qubits (line 124) ---

    #[test]
    fn test_lazy_gate_matrix_1q_wrong_num_qubits() {
        let gate = Arc::new(MockTwoQubitGate) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let result = lazy.matrix_1q();
        assert!(result.is_err(), "Expected error for 2-qubit gate passed to matrix_1q");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("single-qubit"),
            "Error should mention single-qubit: {}",
            err_msg
        );
    }

    // --- LazyGate::matrix_1q() error: gate returns None matrix (lines 134-138) ---

    #[test]
    fn test_lazy_gate_matrix_1q_no_matrix() {
        let gate = Arc::new(MockNoMatrixGate) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let result = lazy.matrix_1q();
        assert!(result.is_err(), "Expected error when gate has no matrix");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("does not provide a matrix"), "Got: {}", err_msg);
    }

    // --- LazyGate::matrix_1q() error: wrong matrix size (lines 141-144) ---

    #[test]
    fn test_lazy_gate_matrix_1q_bad_matrix_size() {
        let gate = Arc::new(MockBadMatrix1qGate) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let result = lazy.matrix_1q();
        assert!(result.is_err(), "Expected error when matrix has wrong size");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("2x2 matrix"), "Got: {}", err_msg);
    }

    // --- LazyGate::matrix_2q() happy path with caching (lines 160-202) ---

    #[test]
    fn test_lazy_gate_matrix_2q_success_and_cache() {
        let gate = Arc::new(CNot) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);

        // matrix_2q should not be computed yet
        assert!(lazy.matrix_2q.borrow().is_none());

        // First call computes and caches
        let m1 = lazy.matrix_2q().unwrap();
        assert!(lazy.matrix_2q.borrow().is_some());

        // Second call returns cached value
        let m2 = lazy.matrix_2q().unwrap();
        assert_eq!(m1, m2);
    }

    // --- LazyGate::matrix_2q() error: wrong num_qubits (line 162) ---

    #[test]
    fn test_lazy_gate_matrix_2q_wrong_num_qubits() {
        let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let result = lazy.matrix_2q();
        assert!(result.is_err(), "Expected error for 1-qubit gate passed to matrix_2q");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("two-qubit"), "Got: {}", err_msg);
    }

    // --- LazyGate::matrix_2q() error: gate returns None matrix (lines 171-176) ---

    #[test]
    fn test_lazy_gate_matrix_2q_no_matrix() {
        let gate = Arc::new(MockNoMatrix2qGate) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let result = lazy.matrix_2q();
        assert!(result.is_err(), "Expected error when 2q gate has no matrix");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("does not provide a matrix"), "Got: {}", err_msg);
    }

    // --- LazyGate::matrix_2q() error: wrong matrix size (lines 178-183) ---

    #[test]
    fn test_lazy_gate_matrix_2q_bad_matrix_size() {
        let gate = Arc::new(MockBadMatrix2qGate) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let result = lazy.matrix_2q();
        assert!(result.is_err(), "Expected error when 2q matrix has wrong size");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("4x4 matrix"), "Got: {}", err_msg);
    }

    // --- LazyGate Debug impl (lines 205-213) ---

    #[test]
    fn test_lazy_gate_debug() {
        let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let lazy = LazyGate::new(gate);
        let debug_str = format!("{:?}", lazy);
        assert!(debug_str.contains("LazyGate"), "Got: {}", debug_str);
        assert!(debug_str.contains("matrix_1q_computed"), "Got: {}", debug_str);
    }

    // --- MatrixCacheKey via get_or_compute_2q (lines 215-250) ---

    #[test]
    fn test_matrix_cache_2q_hit_and_miss() {
        let mut cache = MatrixCache::new(10);
        let cnot_gate = Arc::new(CNot) as Arc<dyn Gate>;

        // First access: miss → compute
        let m1 = cache.get_or_compute_2q(&cnot_gate).unwrap();
        assert_eq!(cache.stats().entries_2q, 1);

        // Second access: hit → return cached
        let m2 = cache.get_or_compute_2q(&cnot_gate).unwrap();
        assert_eq!(m1, m2);
    }

    // --- MatrixCache::get_or_compute_1q error: gate returns no matrix ---

    #[test]
    fn test_matrix_cache_1q_no_matrix_error() {
        let mut cache = MatrixCache::new(10);
        let gate = Arc::new(MockNoMatrixGate) as Arc<dyn Gate>;
        let result = cache.get_or_compute_1q(&gate);
        assert!(result.is_err());
    }

    // --- MatrixCache::get_or_compute_1q error: wrong matrix size ---

    #[test]
    fn test_matrix_cache_1q_bad_matrix_size_error() {
        let mut cache = MatrixCache::new(10);
        let gate = Arc::new(MockBadMatrix1qGate) as Arc<dyn Gate>;
        let result = cache.get_or_compute_1q(&gate);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("2x2 matrix"), "Got: {}", err_msg);
    }

    // --- MatrixCache::get_or_compute_2q error: gate returns no matrix ---

    #[test]
    fn test_matrix_cache_2q_no_matrix_error() {
        let mut cache = MatrixCache::new(10);
        let gate = Arc::new(MockNoMatrix2qGate) as Arc<dyn Gate>;
        let result = cache.get_or_compute_2q(&gate);
        assert!(result.is_err());
    }

    // --- MatrixCache::get_or_compute_2q error: wrong matrix size ---

    #[test]
    fn test_matrix_cache_2q_bad_matrix_size_error() {
        let mut cache = MatrixCache::new(10);
        let gate = Arc::new(MockBadMatrix2qGate) as Arc<dyn Gate>;
        let result = cache.get_or_compute_2q(&gate);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("4x4 matrix"), "Got: {}", err_msg);
    }

    // --- MatrixCache LRU eviction for 2q cache ---

    #[test]
    fn test_matrix_cache_2q_lru_eviction() {
        // max_size=1 so adding a second 2q entry triggers eviction
        let mut cache = MatrixCache::new(1);
        let cnot_gate = Arc::new(CNot) as Arc<dyn Gate>;
        let mock_gate = Arc::new(MockTwoQubitGate) as Arc<dyn Gate>;

        cache.get_or_compute_2q(&cnot_gate).unwrap();
        assert_eq!(cache.stats().total_entries, 1);

        // Adding second entry triggers eviction of first
        cache.get_or_compute_2q(&mock_gate).unwrap();
        assert_eq!(cache.stats().total_entries, 1);
    }

    // --- MatrixCache::clear ---

    #[test]
    fn test_matrix_cache_clear() {
        let mut cache = MatrixCache::new(10);
        let h_gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        let cnot_gate = Arc::new(CNot) as Arc<dyn Gate>;

        cache.get_or_compute_1q(&h_gate).unwrap();
        cache.get_or_compute_2q(&cnot_gate).unwrap();
        assert_eq!(cache.stats().total_entries, 2);

        cache.clear();
        assert_eq!(cache.stats().total_entries, 0);
    }

    // --- CacheStats Display (lines 410-418) ---

    #[test]
    fn test_cache_stats_display() {
        let mut cache = MatrixCache::new(10);
        let h_gate = Arc::new(Hadamard) as Arc<dyn Gate>;
        cache.get_or_compute_1q(&h_gate).unwrap();
        let stats = cache.stats();
        let display = format!("{}", stats);
        assert!(display.contains("Cache:"), "Got: {}", display);
    }

    // --- ExecutorStats::cache_hit_rate zero-total branch ---

    #[test]
    fn test_executor_stats_zero_hit_rate() {
        let stats = ExecutorStats::default();
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }

    // --- ExecutorStats Display ---

    #[test]
    fn test_executor_stats_display() {
        let stats = ExecutorStats {
            cache_hits: 3,
            cache_misses: 1,
            matrices_computed: 1,
            gates_fused: 0,
            batches_executed: 2,
        };
        let display = format!("{}", stats);
        assert!(display.contains("Executor Stats:"), "Got: {}", display);
    }

    // --- LazyExecutor Debug impl ---

    #[test]
    fn test_lazy_executor_debug() {
        let executor = LazyExecutor::default();
        let debug_str = format!("{:?}", executor);
        assert!(debug_str.contains("LazyExecutor"), "Got: {}", debug_str);
    }

    // -----------------------------------------------------------------------
    // New tests to cover LazyExecutor::execute paths
    // -----------------------------------------------------------------------

    /// A mock 3-qubit gate for testing the unsupported gate error path.
    #[derive(Debug)]
    struct MockThreeQubitGate;

    impl Gate for MockThreeQubitGate {
        fn name(&self) -> &str {
            "MockThreeQubit"
        }

        fn num_qubits(&self) -> usize {
            3
        }

        fn matrix(&self) -> Option<Vec<Complex64>> {
            Some(vec![Complex64::new(1.0, 0.0); 64])
        }
    }

    /// Build an initial |0…0⟩ state vector for n qubits.
    fn zero_state(n: usize) -> Vec<Complex64> {
        let mut v = vec![Complex64::new(0.0, 0.0); 1 << n];
        v[0] = Complex64::new(1.0, 0.0);
        v
    }

    /// Lines 497-501: state size mismatch returns ValidationError.
    #[test]
    fn test_lazy_executor_state_size_mismatch() {
        let mut executor = LazyExecutor::default();
        let circuit = Circuit::new(2); // expects 4-element state
                                       // Provide an 8-element state → mismatch
        let mut state = zero_state(3);
        let result = executor.execute(&circuit, &mut state);
        assert!(result.is_err(), "Expected error on size mismatch");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("State size") || err.contains("state"), "Got: {}", err);
    }

    /// Line 600: cache hit on 2-qubit gate (second call to execute with same circuit).
    #[test]
    fn test_lazy_executor_2q_cache_hit() {
        use ferriq_core::QubitId;
        let config = LazyConfig {
            enable_caching: true,
            enable_fusion: false, // disable fusion to keep circuit unchanged
            ..LazyConfig::default()
        };
        let mut executor = LazyExecutor::new(config);

        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(CNot) as Arc<dyn Gate>, &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        // First execution: cache miss
        let mut state1 = zero_state(2);
        executor.execute(&circuit, &mut state1).unwrap();
        assert_eq!(executor.stats().cache_misses, 1);

        // Second execution: cache hit (line 600)
        let mut state2 = zero_state(2);
        executor.execute(&circuit, &mut state2).unwrap();
        assert!(
            executor.stats().cache_hits >= 1,
            "Expected at least one cache hit on second execution"
        );
    }

    /// Lines 608-610: 2-qubit gate executed without caching.
    #[test]
    fn test_lazy_executor_2q_no_caching() {
        use ferriq_core::QubitId;
        let config = LazyConfig {
            enable_caching: false,
            enable_fusion: false,
            ..LazyConfig::default()
        };
        let mut executor = LazyExecutor::new(config);

        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(CNot) as Arc<dyn Gate>, &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let mut state = zero_state(2);
        let result = executor.execute(&circuit, &mut state);
        assert!(result.is_ok());
        // With no caching, matrices_computed should be 1 (computed directly each time)
        assert_eq!(executor.stats().matrices_computed, 1);
    }

    /// Lines 622-625: 3-qubit gate triggers unsupported-gate error.
    #[test]
    fn test_lazy_executor_3q_gate_error() {
        use ferriq_core::QubitId;
        let config = LazyConfig {
            enable_fusion: false,
            ..LazyConfig::default()
        };
        let mut executor = LazyExecutor::new(config);

        let mut circuit = Circuit::new(3);
        circuit
            .add_gate(
                Arc::new(MockThreeQubitGate) as Arc<dyn Gate>,
                &[QubitId::new(0), QubitId::new(1), QubitId::new(2)],
            )
            .unwrap();

        let mut state = zero_state(3);
        let result = executor.execute(&circuit, &mut state);
        assert!(result.is_err(), "Expected error for 3-qubit gate");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("3") || err.contains("qubit"), "Got: {}", err);
    }
}
