//! Gate matrix caching

use crate::execution_engine::kernels::{Matrix2x2, Matrix4x4, Matrix8x8};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Cache key for gate matrices
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct GateCacheKey {
    pub gate_name: String,
    pub params: Vec<OrderedFloat>,
}

/// Wrapper for f64 that implements Eq and Hash
#[derive(Debug, Clone, Copy)]
pub struct OrderedFloat(pub f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Cached gate matrix
#[derive(Debug, Clone)]
pub enum CachedMatrix {
    Single(Matrix2x2),
    Two(Matrix4x4),
    Three(Box<Matrix8x8>),
}

/// LRU cache for gate matrices
pub struct GateMatrixCache {
    cache: RwLock<HashMap<GateCacheKey, Arc<CachedMatrix>>>,
    max_size: usize,
    hits: parking_lot::RwLock<usize>,
    misses: parking_lot::RwLock<usize>,
}

impl GateMatrixCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_size,
            hits: parking_lot::RwLock::new(0),
            misses: parking_lot::RwLock::new(0),
        }
    }

    pub fn get(&self, key: &GateCacheKey) -> Option<Arc<CachedMatrix>> {
        let result = self.cache.read().get(key).cloned();
        if result.is_some() {
            *self.hits.write() += 1;
        } else {
            *self.misses.write() += 1;
        }
        result
    }

    pub fn insert(&self, key: GateCacheKey, matrix: CachedMatrix) {
        let mut cache = self.cache.write();
        if cache.len() >= self.max_size {
            // Simple eviction: remove first entry (not truly LRU, but simple)
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        cache.insert(key, Arc::new(matrix));
    }

    pub fn clear(&self) {
        self.cache.write().clear();
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hits.read();
        let misses = *self.misses.read();
        let total = hits + misses;
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl Default for GateMatrixCache {
    fn default() -> Self {
        Self::new(128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn make_single_matrix() -> CachedMatrix {
        CachedMatrix::Single([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
    }

    #[test]
    fn test_default_cache() {
        let cache = GateMatrixCache::default();
        assert_eq!(cache.max_size, 128);
    }

    #[test]
    fn test_cache_eviction_when_full() {
        // Create a cache with max_size=2
        let cache = GateMatrixCache::new(2);

        let key1 = GateCacheKey {
            gate_name: "H".to_string(),
            params: vec![],
        };
        let key2 = GateCacheKey {
            gate_name: "X".to_string(),
            params: vec![],
        };
        let key3 = GateCacheKey {
            gate_name: "Z".to_string(),
            params: vec![],
        };

        cache.insert(key1, make_single_matrix());
        cache.insert(key2, make_single_matrix());
        // Inserting a third entry should evict one
        cache.insert(key3.clone(), make_single_matrix());

        // Cache should still have at most 2 entries
        assert!(cache.cache.read().len() <= 2);
        // The newly inserted entry should be present
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_hit_rate() {
        let cache = GateMatrixCache::new(10);
        assert_eq!(cache.hit_rate(), 0.0); // no accesses yet

        let key = GateCacheKey {
            gate_name: "H".to_string(),
            params: vec![],
        };
        cache.insert(key.clone(), make_single_matrix());
        let _ = cache.get(&key); // hit
        let miss_key = GateCacheKey {
            gate_name: "X".to_string(),
            params: vec![],
        };
        let _ = cache.get(&miss_key); // miss

        let rate = cache.hit_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ordered_float_hash_eq() {
        use std::collections::HashSet;
        let f1 = OrderedFloat(1.0);
        let f2 = OrderedFloat(1.0);
        let f3 = OrderedFloat(2.0);

        assert_eq!(f1, f2);
        assert_ne!(f1, f3);

        let mut set = HashSet::new();
        set.insert(f1);
        set.insert(f2); // duplicate
        set.insert(f3);
        assert_eq!(set.len(), 2);
    }
}
