//! Topology-keyed cache for multi-qubit fusion block *structure*.
//!
//! Caches which circuit operations group into which
//! [`crate::fusion::FusionBlock`]s (see `fusion.rs`), keyed by
//! [`crate::cache::CircuitFingerprint`] — a structural hash of gate names
//! and qubit targets that, by its own documented contract, "doesn't account
//! for parameter values in parameterized gates." Motivation: a VQE/QAOA
//! outer loop recompiles the *same-shaped* circuit hundreds of times with
//! only rotation angles differing (see BENCHMARKS.md's own methodology
//! note: "One benchmark iteration = one full cost-function evaluation").
//!
//! **Why caching block *structure* (not the compiled circuit) is safe.**
//! `crate::cached_compiler::CachedCompiler` already caches *entire compiled
//! circuits* by this same fingerprint — which means a cache hit there
//! returns a previous call's *baked-in matrices*, silently wrong for a
//! parametrized gate whose angle changed between calls. This module
//! deliberately caches at a different granularity: [`find_fusion_blocks`]
//! (see `fusion.rs`) only ever reads `gate.name()`, `op.qubits()`, and
//! whether `gate.matrix().is_some()` — never a concrete matrix element — so
//! its *output* (which operation indices group into which qubit-ordered
//! block) is providably identical for any two circuits with the same
//! fingerprint, regardless of parameter values. A cache hit here means
//! "this call would have produced the exact same block structure," not
//! "probably similar." The actual fused matrix is still recomputed fresh
//! from each call's concrete component gate matrices — see
//! `fusion.rs::fuse_multi_qubit_blocks`.
//!
//! Bounded, LRU-evicted, `Mutex`-guarded — mirrors
//! [`crate::cache::CompilationCache`]/[`crate::cache::SharedCompilationCache`]'s
//! existing pattern rather than introducing a new caching idiom.
//!
//! Only ever consulted from the multi-qubit block path
//! (`FusionConfig::max_block_width > 1`, i.e.
//! `circuit.num_qubits() >= FusionConfig::parallel_threshold_qubits`) — see
//! `fusion.rs`'s dispatch. Below that threshold this cache is never
//! touched, consistent with keeping the small-circuit path's cost at
//! exactly what it was before this feature.

use crate::cache::CircuitFingerprint;
use crate::fusion::FusionBlock;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// LRU cache from a circuit's [`CircuitFingerprint`] to the fusion block
/// structure `find_fusion_blocks` would produce for it. See module docs for
/// why this is exact, not approximate.
#[derive(Debug)]
pub struct FusionStructureCache {
    max_size: usize,
    cache: Mutex<FusionStructureCacheInner>,
}

#[derive(Debug)]
struct FusionStructureCacheInner {
    entries: HashMap<CircuitFingerprint, Arc<Vec<FusionBlock>>>,
    lru_queue: VecDeque<CircuitFingerprint>,
    hits: usize,
    misses: usize,
}

impl FusionStructureCache {
    /// Create a new cache holding at most `max_size` distinct circuit
    /// shapes (0 = caching disabled: every call is a miss, nothing is
    /// stored).
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            cache: Mutex::new(FusionStructureCacheInner {
                entries: HashMap::new(),
                lru_queue: VecDeque::new(),
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Look up the fusion block structure for a circuit fingerprint.
    /// Moves the entry to the back of the LRU queue on a hit.
    pub(crate) fn get(&self, fingerprint: CircuitFingerprint) -> Option<Arc<Vec<FusionBlock>>> {
        let mut inner = self.cache.lock().unwrap();
        if let Some(blocks) = inner.entries.get(&fingerprint).cloned() {
            inner.lru_queue.retain(|fp| *fp != fingerprint);
            inner.lru_queue.push_back(fingerprint);
            inner.hits += 1;
            Some(blocks)
        } else {
            inner.misses += 1;
            None
        }
    }

    /// Insert a freshly-computed block structure, evicting the least
    /// recently used entry if the cache is full.
    pub(crate) fn insert(&self, fingerprint: CircuitFingerprint, blocks: Vec<FusionBlock>) {
        if self.max_size == 0 {
            return;
        }
        let mut inner = self.cache.lock().unwrap();
        if inner.entries.len() >= self.max_size && !inner.entries.contains_key(&fingerprint) {
            if let Some(lru_fp) = inner.lru_queue.pop_front() {
                inner.entries.remove(&lru_fp);
            }
        }
        inner.lru_queue.retain(|fp| *fp != fingerprint);
        inner.lru_queue.push_back(fingerprint);
        inner.entries.insert(fingerprint, Arc::new(blocks));
    }

    /// Number of cache hits since creation (or the last [`Self::clear`]).
    pub fn hits(&self) -> usize {
        self.cache.lock().unwrap().hits
    }

    /// Number of cache misses since creation (or the last [`Self::clear`]).
    pub fn misses(&self) -> usize {
        self.cache.lock().unwrap().misses
    }

    /// Number of distinct circuit shapes currently cached.
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().entries.len()
    }

    /// Whether the cache is currently empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries and reset hit/miss counters.
    pub fn clear(&self) {
        let mut inner = self.cache.lock().unwrap();
        inner.entries.clear();
        inner.lru_queue.clear();
        inner.hits = 0;
        inner.misses = 0;
    }
}

impl Default for FusionStructureCache {
    /// Matches `CompilationCache`'s default size (100) — a VQE/QAOA loop
    /// only ever needs one entry per distinct circuit *shape* it evaluates,
    /// so this comfortably covers realistic outer loops without unbounded
    /// growth.
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::{find_fusion_blocks, FusionConfig};
    use simq_core::{gate::Gate, Circuit, QubitId};
    use std::sync::Arc as StdArc;

    #[derive(Debug)]
    struct MockGate {
        name: String,
        num_qubits: usize,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }
        fn num_qubits(&self) -> usize {
            self.num_qubits
        }
        fn matrix(&self) -> Option<Vec<num_complex::Complex64>> {
            let dim = 1usize << self.num_qubits;
            let mut m = vec![num_complex::Complex64::new(0.0, 0.0); dim * dim];
            for i in 0..dim {
                m[i * dim + i] = num_complex::Complex64::new(1.0, 0.0);
            }
            Some(m)
        }
    }

    fn sample_circuit() -> Circuit {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        circuit
            .add_gate(
                StdArc::new(MockGate {
                    name: "H".to_string(),
                    num_qubits: 1,
                }) as StdArc<dyn Gate>,
                &[q0],
            )
            .unwrap();
        circuit
            .add_gate(
                StdArc::new(MockGate {
                    name: "CNOT".to_string(),
                    num_qubits: 2,
                }) as StdArc<dyn Gate>,
                &[q0, q1],
            )
            .unwrap();
        circuit
    }

    #[test]
    fn test_cache_miss_then_hit() {
        let cache = FusionStructureCache::new(10);
        let circuit = sample_circuit();
        let fp = CircuitFingerprint::compute(&circuit);

        assert!(cache.get(fp).is_none());
        assert_eq!(cache.misses(), 1);

        let config = FusionConfig::default();
        let blocks = find_fusion_blocks(&circuit, &config);
        cache.insert(fp, blocks);

        assert!(cache.get(fp).is_some());
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = FusionStructureCache::new(2);
        let mut fps = vec![];
        for i in 0..3 {
            let mut circuit = Circuit::new(i + 2);
            circuit
                .add_gate(
                    StdArc::new(MockGate {
                        name: format!("G{}", i),
                        num_qubits: 1,
                    }) as StdArc<dyn Gate>,
                    &[QubitId::new(0)],
                )
                .unwrap();
            let fp = CircuitFingerprint::compute(&circuit);
            cache.insert(fp, vec![]);
            fps.push(fp);
        }

        assert_eq!(cache.len(), 2);
        // First entry should have been evicted (LRU).
        assert!(cache.get(fps[0]).is_none());
        assert!(cache.get(fps[1]).is_some());
        assert!(cache.get(fps[2]).is_some());
    }

    #[test]
    fn test_zero_size_disables_caching() {
        let cache = FusionStructureCache::new(0);
        let circuit = sample_circuit();
        let fp = CircuitFingerprint::compute(&circuit);

        cache.insert(fp, vec![]);
        assert!(cache.get(fp).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn test_clear_resets_counters_and_entries() {
        let cache = FusionStructureCache::new(10);
        let circuit = sample_circuit();
        let fp = CircuitFingerprint::compute(&circuit);
        cache.insert(fp, vec![]);
        cache.get(fp);
        cache.get(CircuitFingerprint::compute(&Circuit::new(5)));

        assert!(cache.hits() > 0);
        assert!(cache.misses() > 0);

        cache.clear();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_default_size_matches_compilation_cache() {
        let cache = FusionStructureCache::default();
        assert!(cache.is_empty());
        // Fill past 100 to confirm the default bound is enforced.
        for i in 0..105 {
            let mut circuit = Circuit::new(2);
            circuit
                .add_gate(
                    StdArc::new(MockGate {
                        name: format!("G{}", i),
                        num_qubits: 1,
                    }) as StdArc<dyn Gate>,
                    &[QubitId::new(0)],
                )
                .unwrap();
            cache.insert(CircuitFingerprint::compute(&circuit), vec![]);
        }
        assert_eq!(cache.len(), 100);
    }
}
