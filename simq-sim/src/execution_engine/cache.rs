//! Gate matrix caching

use num_complex::Complex64;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use crate::execution_engine::kernels::{Matrix2x2, Matrix4x4};

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
