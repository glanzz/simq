//! Compilation caching
//!
//! This module provides caching mechanisms to avoid redundant compilation
//! of identical or similar circuits. Caching can dramatically improve
//! compilation performance in scenarios with repeated patterns.

use simq_core::Circuit;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// A fingerprint uniquely identifying a circuit for caching purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CircuitFingerprint(u64);

impl CircuitFingerprint {
    /// Compute a fingerprint from a circuit
    ///
    /// The fingerprint is computed from:
    /// - Number of qubits
    /// - Gate sequence (gate names and qubit targets)
    ///
    /// Note: This is a structural fingerprint that doesn't account for
    /// parameter values in parameterized gates. For parameter-sensitive
    /// caching, a more sophisticated approach would be needed.
    pub fn compute(circuit: &Circuit) -> Self {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash number of qubits
        circuit.num_qubits().hash(&mut hasher);

        // Hash gate count
        circuit.len().hash(&mut hasher);

        // Hash each gate in sequence
        for op in circuit.operations() {
            // Hash gate name
            op.gate().name().hash(&mut hasher);

            // Hash qubit targets
            for qubit in op.qubits() {
                qubit.index().hash(&mut hasher);
            }
        }

        CircuitFingerprint(hasher.finish())
    }

    /// Get the raw hash value
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Statistics about cache performance
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Number of cache evictions
    pub evictions: usize,
    /// Current cache size
    pub current_size: usize,
    /// Maximum cache size
    pub max_size: usize,
}

impl CacheStatistics {
    /// Calculate hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }

    /// Calculate miss rate as a percentage
    pub fn miss_rate(&self) -> f64 {
        100.0 - self.hit_rate()
    }
}

impl std::fmt::Display for CacheStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cache Statistics:")?;
        writeln!(f, "  Hits: {}", self.hits)?;
        writeln!(f, "  Misses: {}", self.misses)?;
        writeln!(f, "  Hit rate: {:.1}%", self.hit_rate())?;
        writeln!(f, "  Evictions: {}", self.evictions)?;
        writeln!(f, "  Current size: {}/{}", self.current_size, self.max_size)?;
        Ok(())
    }
}

/// LRU cache for compiled circuits
///
/// This cache stores optimized circuits indexed by their fingerprint,
/// using a Least Recently Used (LRU) eviction policy.
pub struct CompilationCache {
    /// Maximum number of entries in the cache
    max_size: usize,
    /// Cache storage: fingerprint -> optimized circuit
    cache: HashMap<CircuitFingerprint, Circuit>,
    /// LRU queue: most recently used at the back
    lru_queue: VecDeque<CircuitFingerprint>,
    /// Cache statistics
    stats: CacheStatistics,
}

impl CompilationCache {
    /// Create a new compilation cache with the specified maximum size
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of circuits to cache (0 = unlimited)
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            cache: HashMap::new(),
            lru_queue: VecDeque::new(),
            stats: CacheStatistics {
                max_size,
                ..Default::default()
            },
        }
    }

    /// Create a cache with default size (100 entries)
    pub fn default() -> Self {
        Self::new(100)
    }

    /// Get a cached circuit by fingerprint
    ///
    /// If found, moves the entry to the back of the LRU queue.
    pub fn get(&mut self, fingerprint: CircuitFingerprint) -> Option<Circuit> {
        if let Some(circuit) = self.cache.get(&fingerprint) {
            // Move to back of LRU queue (most recently used)
            self.lru_queue.retain(|fp| *fp != fingerprint);
            self.lru_queue.push_back(fingerprint);

            self.stats.hits += 1;
            Some(circuit.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Insert a circuit into the cache
    ///
    /// If the cache is full, evicts the least recently used entry.
    pub fn insert(&mut self, fingerprint: CircuitFingerprint, circuit: Circuit) {
        // Check if we need to evict
        if self.max_size > 0
            && self.cache.len() >= self.max_size
            && !self.cache.contains_key(&fingerprint)
        {
            if let Some(lru_fingerprint) = self.lru_queue.pop_front() {
                self.cache.remove(&lru_fingerprint);
                self.stats.evictions += 1;
            }
        }

        // Insert or update
        match self.cache.entry(fingerprint) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                // Update existing entry
                e.insert(circuit);
                // Move to back of queue
                self.lru_queue.retain(|fp| *fp != fingerprint);
                self.lru_queue.push_back(fingerprint);
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                // Insert new entry
                e.insert(circuit);
                self.lru_queue.push_back(fingerprint);
            }
        }

        self.stats.current_size = self.cache.len();
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_queue.clear();
        self.stats.evictions += self.stats.current_size;
        self.stats.current_size = 0;
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.stats.clone()
    }

    /// Get the current cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get the maximum cache size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Set the maximum cache size
    ///
    /// If the new size is smaller than the current cache size,
    /// evicts entries until the size constraint is met.
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
        self.stats.max_size = max_size;

        // Evict if necessary
        while max_size > 0 && self.cache.len() > max_size {
            if let Some(lru_fingerprint) = self.lru_queue.pop_front() {
                self.cache.remove(&lru_fingerprint);
                self.stats.evictions += 1;
            }
        }

        self.stats.current_size = self.cache.len();
    }
}

/// Thread-safe compilation cache
///
/// Wraps CompilationCache in Arc<Mutex<>> for concurrent access.
#[derive(Clone)]
pub struct SharedCompilationCache {
    inner: Arc<Mutex<CompilationCache>>,
}

impl SharedCompilationCache {
    /// Create a new shared cache
    pub fn new(max_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CompilationCache::new(max_size))),
        }
    }

    /// Create a shared cache with default size
    pub fn default() -> Self {
        Self::new(100)
    }

    /// Get a cached circuit
    pub fn get(&self, fingerprint: CircuitFingerprint) -> Option<Circuit> {
        self.inner.lock().unwrap().get(fingerprint)
    }

    /// Insert a circuit into the cache
    pub fn insert(&self, fingerprint: CircuitFingerprint, circuit: Circuit) {
        self.inner.lock().unwrap().insert(fingerprint, circuit);
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.inner.lock().unwrap().clear();
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.inner.lock().unwrap().statistics()
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// Set maximum cache size
    pub fn set_max_size(&self, max_size: usize) {
        self.inner.lock().unwrap().set_max_size(max_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_fingerprint_identical_circuits() {
        let mut circuit1 = Circuit::new(2);
        let mut circuit2 = Circuit::new(2);

        let gate = Arc::new(MockGate {
            name: "H".to_string(),
        });

        circuit1.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();
        circuit2.add_gate(gate.clone(), &[QubitId::new(0)]).unwrap();

        let fp1 = CircuitFingerprint::compute(&circuit1);
        let fp2 = CircuitFingerprint::compute(&circuit2);

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_circuits() {
        let mut circuit1 = Circuit::new(2);
        let mut circuit2 = Circuit::new(2);

        let h = Arc::new(MockGate {
            name: "H".to_string(),
        });
        let x = Arc::new(MockGate {
            name: "X".to_string(),
        });

        circuit1.add_gate(h.clone(), &[QubitId::new(0)]).unwrap();
        circuit2.add_gate(x.clone(), &[QubitId::new(0)]).unwrap();

        let fp1 = CircuitFingerprint::compute(&circuit1);
        let fp2 = CircuitFingerprint::compute(&circuit2);

        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = CompilationCache::new(10);

        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let fp = CircuitFingerprint::compute(&circuit);

        // First access should miss
        assert!(cache.get(fp).is_none());
        assert_eq!(cache.statistics().hits, 0);
        assert_eq!(cache.statistics().misses, 1);

        // Insert
        cache.insert(fp, circuit.clone());

        // Second access should hit
        assert!(cache.get(fp).is_some());
        assert_eq!(cache.statistics().hits, 1);
        assert_eq!(cache.statistics().misses, 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = CompilationCache::new(2);

        // Create three different circuits
        let mut circuits: Vec<Circuit> = vec![];
        for i in 0..3 {
            let mut circuit = Circuit::new(i + 2); // Different number of qubits
            let gate = Arc::new(MockGate {
                name: format!("G{}", i), // Different gate names
            });
            circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();
            circuits.push(circuit);
        }

        let fps: Vec<_> = circuits.iter().map(CircuitFingerprint::compute).collect();

        // Verify fingerprints are unique
        assert_ne!(fps[0], fps[1]);
        assert_ne!(fps[1], fps[2]);
        assert_ne!(fps[0], fps[2]);

        // Fill cache
        cache.insert(fps[0], circuits[0].clone());
        cache.insert(fps[1], circuits[1].clone());
        assert_eq!(cache.len(), 2);

        // Insert third should evict first (LRU)
        cache.insert(fps[2], circuits[2].clone());
        assert_eq!(cache.len(), 2);

        // First should be evicted
        assert!(cache.get(fps[0]).is_none());
        // Second and third should still be present
        assert!(cache.get(fps[1]).is_some());
        assert!(cache.get(fps[2]).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = CompilationCache::new(10);
        let circuit = Circuit::new(2);
        let fp = CircuitFingerprint::compute(&circuit);

        cache.insert(fp, circuit);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_resize() {
        let mut cache = CompilationCache::new(5);

        // Fill cache
        for i in 0..5 {
            let circuit = Circuit::new(i + 1);
            let fp = CircuitFingerprint::compute(&circuit);
            cache.insert(fp, circuit);
        }
        assert_eq!(cache.len(), 5);

        // Resize to smaller
        cache.set_max_size(3);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.statistics().evictions, 2);
    }

    #[test]
    fn test_shared_cache() {
        let cache = SharedCompilationCache::new(10);

        let mut circuit = Circuit::new(2);
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
        });
        circuit.add_gate(gate, &[QubitId::new(0)]).unwrap();

        let fp = CircuitFingerprint::compute(&circuit);

        // Insert and retrieve
        cache.insert(fp, circuit.clone());
        assert!(cache.get(fp).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_statistics() {
        let mut cache = CompilationCache::new(2);

        let circuit = Circuit::new(2);
        let fp = CircuitFingerprint::compute(&circuit);

        // Miss
        cache.get(fp);
        let stats = cache.statistics();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 1);

        // Insert and hit
        cache.insert(fp, circuit);
        cache.get(fp);
        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate(), 50.0);
    }
}
