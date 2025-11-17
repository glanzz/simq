//! Cached compilation
//!
//! This module provides a caching wrapper around the compiler that
//! automatically caches compilation results.

use crate::cache::{CircuitFingerprint, CompilationCache, SharedCompilationCache};
use crate::compiler::Compiler;
use crate::passes::OptimizationResult;
use simq_core::{Circuit, Result};

/// A compiler that caches compilation results
///
/// This wrapper automatically caches optimized circuits, avoiding
/// redundant compilation of identical circuits.
pub struct CachedCompiler {
    /// The underlying compiler
    compiler: Compiler,
    /// The compilation cache
    cache: CompilationCache,
    /// Whether caching is enabled
    enabled: bool,
}

impl CachedCompiler {
    /// Create a new cached compiler
    ///
    /// # Arguments
    /// * `compiler` - The compiler to wrap
    /// * `cache_size` - Maximum number of circuits to cache
    pub fn new(compiler: Compiler, cache_size: usize) -> Self {
        Self {
            compiler,
            cache: CompilationCache::new(cache_size),
            enabled: true,
        }
    }

    /// Create a cached compiler with default cache size (100)
    pub fn with_compiler(compiler: Compiler) -> Self {
        Self::new(compiler, 100)
    }

    /// Enable or disable caching
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Compile a circuit with caching
    ///
    /// If the circuit has been compiled before (same fingerprint),
    /// returns the cached result. Otherwise, compiles and caches.
    ///
    /// # Arguments
    /// * `circuit` - The circuit to compile (modified in-place)
    ///
    /// # Returns
    /// * `Ok(OptimizationResult)` with compilation statistics
    /// * The `cached` field indicates whether the result was from cache
    pub fn compile(&mut self, circuit: &mut Circuit) -> Result<CachedOptimizationResult> {
        if !self.enabled {
            // Caching disabled, compile directly
            let result = self.compiler.compile(circuit)?;
            return Ok(CachedOptimizationResult {
                inner: result,
                cached: false,
            });
        }

        // Compute fingerprint of input circuit
        let fingerprint = CircuitFingerprint::compute(circuit);

        // Check cache
        if let Some(cached_circuit) = self.cache.get(fingerprint) {
            // Cache hit! Replace circuit with cached version
            *circuit = cached_circuit;
            return Ok(CachedOptimizationResult {
                inner: OptimizationResult::new(),
                cached: true,
            });
        }

        // Cache miss - compile the circuit
        let result = self.compiler.compile(circuit)?;

        // Store in cache
        self.cache.insert(fingerprint, circuit.clone());

        Ok(CachedOptimizationResult {
            inner: result,
            cached: false,
        })
    }

    /// Get a reference to the cache
    pub fn cache(&self) -> &CompilationCache {
        &self.cache
    }

    /// Get a mutable reference to the cache
    pub fn cache_mut(&mut self) -> &mut CompilationCache {
        &mut self.cache
    }

    /// Clear the compilation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the underlying compiler
    pub fn compiler(&self) -> &Compiler {
        &self.compiler
    }
}

/// Result of a cached compilation
#[derive(Debug, Clone)]
pub struct CachedOptimizationResult {
    /// The optimization result (if not from cache)
    pub inner: OptimizationResult,
    /// Whether this result was retrieved from cache
    pub cached: bool,
}

impl CachedOptimizationResult {
    /// Check if this result was from cache
    pub fn is_cached(&self) -> bool {
        self.cached
    }

    /// Get the optimization result
    pub fn result(&self) -> &OptimizationResult {
        &self.inner
    }
}

/// Thread-safe cached compiler
///
/// This version uses a shared cache that can be used across threads.
#[derive(Clone)]
pub struct SharedCachedCompiler {
    /// The underlying compiler (each thread has its own copy)
    compiler: Compiler,
    /// Shared compilation cache
    cache: SharedCompilationCache,
    /// Whether caching is enabled
    enabled: bool,
}

impl SharedCachedCompiler {
    /// Create a new shared cached compiler
    pub fn new(compiler: Compiler, cache_size: usize) -> Self {
        Self {
            compiler,
            cache: SharedCompilationCache::new(cache_size),
            enabled: true,
        }
    }

    /// Create with default cache size
    pub fn with_compiler(compiler: Compiler) -> Self {
        Self::new(compiler, 100)
    }

    /// Set whether caching is enabled
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Compile a circuit with caching
    pub fn compile(&self, circuit: &mut Circuit) -> Result<CachedOptimizationResult> {
        if !self.enabled {
            let result = self.compiler.compile(circuit)?;
            return Ok(CachedOptimizationResult {
                inner: result,
                cached: false,
            });
        }

        let fingerprint = CircuitFingerprint::compute(circuit);

        // Check cache
        if let Some(cached_circuit) = self.cache.get(fingerprint) {
            *circuit = cached_circuit;
            return Ok(CachedOptimizationResult {
                inner: OptimizationResult::new(),
                cached: true,
            });
        }

        // Compile
        let result = self.compiler.compile(circuit)?;
        self.cache.insert(fingerprint, circuit.clone());

        Ok(CachedOptimizationResult {
            inner: result,
            cached: false,
        })
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> crate::cache::CacheStatistics {
        self.cache.statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{create_compiler, OptimizationLevel};
    use simq_core::{gate::Gate, QubitId};
    use std::sync::Arc;

    #[derive(Debug)]
    struct MockGate {
        name: String,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_cached_compiler_hit() {
        let compiler = create_compiler(OptimizationLevel::O1);
        let mut cached = CachedCompiler::new(compiler, 10);

        // Create two identical circuits
        let mut circuit1 = Circuit::new(2);
        let mut circuit2 = Circuit::new(2);

        let gate = Arc::new(MockGate {
            name: "X".to_string(),
        });

        circuit1.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit1.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();

        circuit2.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit2.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();

        // First compilation should not be cached
        let result1 = cached.compile(&mut circuit1).unwrap();
        assert!(!result1.is_cached());

        // Second compilation of identical circuit should be cached
        let result2 = cached.compile(&mut circuit2).unwrap();
        assert!(result2.is_cached());

        // Verify cache statistics
        let stats = cached.cache().statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cached_compiler_disabled() {
        let compiler = create_compiler(OptimizationLevel::O1);
        let mut cached = CachedCompiler::new(compiler, 10);
        cached.set_enabled(false);

        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "X".to_string(),
        });
        circuit.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();

        // First compile
        let result1 = cached.compile(&mut circuit).unwrap();
        assert!(!result1.is_cached());

        // Second compile should also not be cached (caching disabled)
        let result2 = cached.compile(&mut circuit).unwrap();
        assert!(!result2.is_cached());
    }

    #[test]
    fn test_cache_clear() {
        let compiler = create_compiler(OptimizationLevel::O1);
        let mut cached = CachedCompiler::new(compiler, 10);

        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "X".to_string(),
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        // Compile once
        cached.compile(&mut circuit).unwrap();
        assert_eq!(cached.cache().len(), 1);

        // Clear cache
        cached.clear_cache();
        assert_eq!(cached.cache().len(), 0);

        // Recompile should be a cache miss
        let result = cached.compile(&mut circuit).unwrap();
        assert!(!result.is_cached());
    }

    #[test]
    fn test_shared_cached_compiler() {
        let compiler = create_compiler(OptimizationLevel::O1);
        let cached = SharedCachedCompiler::new(compiler, 10);

        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        // First compile
        let result1 = cached.compile(&mut circuit).unwrap();
        assert!(!result1.is_cached());

        // Second compile should hit cache
        let result2 = cached.compile(&mut circuit).unwrap();
        assert!(result2.is_cached());

        // Verify statistics
        let stats = cached.cache_statistics();
        assert_eq!(stats.hits, 1);
    }
}
