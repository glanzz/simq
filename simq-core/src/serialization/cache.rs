//! Circuit caching infrastructure

#![cfg(feature = "cache")]

use crate::{Circuit, QuantumError, Result};
use dashmap::DashMap;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Cache key based on circuit structure hash
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CircuitKey {
    hash: u64,
}

impl CircuitKey {
    /// Create a cache key from a circuit
    pub fn from_circuit(circuit: &Circuit) -> Self {
        Self {
            hash: circuit.cache_key(),
        }
    }

    /// Create a cache key from a hash
    pub fn from_hash(hash: u64) -> Self {
        Self { hash }
    }

    /// Get the hash value
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

/// Cache statistics
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
}

impl CacheStats {
    /// Get hit rate as a percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Trait for circuit cache implementations
pub trait CircuitCache: Send + Sync {
    /// Get a cached circuit by key
    fn get(&self, key: &CircuitKey) -> Option<Circuit>;

    /// Store a circuit in cache
    fn put(&self, key: CircuitKey, circuit: Circuit) -> Result<()>;

    /// Remove a circuit from cache
    fn remove(&self, key: &CircuitKey) -> Option<Circuit>;

    /// Clear all cached circuits
    fn clear(&self);

    /// Get cache statistics
    fn stats(&self) -> CacheStats;
}

/// In-memory circuit cache using DashMap
pub struct MemoryCache {
    cache: DashMap<CircuitKey, Circuit>,
    stats: dashmap::DashMap<String, u64>, // Simple stats tracking
}

impl MemoryCache {
    /// Create a new in-memory cache
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
            stats: DashMap::new(),
        }
    }

    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: DashMap::with_capacity(capacity),
            stats: DashMap::new(),
        }
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitCache for MemoryCache {
    fn get(&self, key: &CircuitKey) -> Option<Circuit> {
        let result = self.cache.get(key).map(|entry| entry.value().clone());
        
        // Update stats
        if result.is_some() {
            *self.stats.entry("hits".to_string()).or_insert(0) += 1;
        } else {
            *self.stats.entry("misses".to_string()).or_insert(0) += 1;
        }
        
        result
    }

    fn put(&self, key: CircuitKey, circuit: Circuit) -> Result<()> {
        self.cache.insert(key, circuit);
        Ok(())
    }

    fn remove(&self, key: &CircuitKey) -> Option<Circuit> {
        self.cache.remove(key).map(|(_, circuit)| circuit)
    }

    fn clear(&self) {
        self.cache.clear();
        self.stats.clear();
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.stats.get("hits").map(|v| *v.value()).unwrap_or(0),
            misses: self.stats.get("misses").map(|v| *v.value()).unwrap_or(0),
            size: self.cache.len(),
        }
    }
}

/// File-based circuit cache
pub struct FileCache {
    cache_dir: PathBuf,
    ttl: Option<Duration>,
}

impl FileCache {
    /// Create a new file cache
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| QuantumError::CacheError(format!("Failed to create cache directory: {}", e)))?;

        Ok(Self {
            cache_dir,
            ttl: None,
        })
    }

    /// Create with time-to-live
    pub fn with_ttl<P: AsRef<Path>>(cache_dir: P, ttl: Duration) -> Result<Self> {
        let mut cache = Self::new(cache_dir)?;
        cache.ttl = Some(ttl);
        Ok(cache)
    }

    /// Get file path for a cache key
    fn key_to_path(&self, key: &CircuitKey) -> PathBuf {
        // Use hash to create directory structure: cache_dir/ab/cdef...bin
        let hash_str = format!("{:016x}", key.hash());
        let (dir, file) = hash_str.split_at(2);
        self.cache_dir.join(dir).join(format!("{}.bin", file))
    }
}

impl CircuitCache for FileCache {
    fn get(&self, key: &CircuitKey) -> Option<Circuit> {
        let file_path = self.key_to_path(key);

        // Check if file exists
        if let Ok(metadata) = std::fs::metadata(&file_path) {
            // Check TTL if set
            if let Some(ttl) = self.ttl {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(elapsed) = SystemTime::now().duration_since(modified) {
                        if elapsed > ttl {
                            // Expired, remove file
                            let _ = std::fs::remove_file(&file_path);
                            return None;
                        }
                    }
                }
            }

            // Read and deserialize
            if let Ok(bytes) = std::fs::read(&file_path) {
                return Circuit::from_bytes(&bytes).ok();
            }
        }

        None
    }

    fn put(&self, key: CircuitKey, circuit: Circuit) -> Result<()> {
        let file_path = self.key_to_path(&key);

        // Ensure directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| QuantumError::CacheError(format!("Failed to create cache directory: {}", e)))?;
        }

        // Serialize and write
        let bytes = circuit.to_bytes()?;
        std::fs::write(&file_path, bytes)
            .map_err(|e| QuantumError::CacheError(format!("Failed to write cache file: {}", e)))?;

        Ok(())
    }

    fn remove(&self, key: &CircuitKey) -> Option<Circuit> {
        let file_path = self.key_to_path(key);
        if file_path.exists() {
            // Try to read before removing
            let circuit = self.get(key);
            let _ = std::fs::remove_file(&file_path);
            circuit
        } else {
            None
        }
    }

    fn clear(&self) {
        // Remove all files in cache directory
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    let _ = std::fs::remove_file(&path);
                } else if path.is_dir() {
                    // Recursively clear subdirectories
                    let _ = std::fs::remove_dir_all(&path);
                }
            }
        }
    }

    fn stats(&self) -> CacheStats {
        // Count files in cache directory
        let mut size = 0;
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("bin") {
                    size += 1;
                } else if path.is_dir() {
                    // Count files in subdirectories
                    if let Ok(sub_entries) = std::fs::read_dir(&path) {
                        size += sub_entries.count();
                    }
                }
            }
        }

        CacheStats {
            hits: 0,
            misses: 0,
            size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_key() {
        let circuit = Circuit::new(2);
        let key1 = CircuitKey::from_circuit(&circuit);
        let key2 = CircuitKey::from_circuit(&circuit);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_memory_cache() {
        let cache = MemoryCache::new();
        let circuit = Circuit::new(2);
        let key = CircuitKey::from_circuit(&circuit);

        // Cache miss
        assert!(cache.get(&key).is_none());

        // Put in cache
        cache.put(key.clone(), circuit.clone()).unwrap();

        // Cache hit
        let cached = cache.get(&key);
        assert!(cached.is_some());

        // Clear cache
        cache.clear();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = MemoryCache::new();
        let circuit = Circuit::new(2);
        let key = CircuitKey::from_circuit(&circuit);

        // Miss
        cache.get(&key);
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Put and hit
        cache.put(key.clone(), circuit).unwrap();
        cache.get(&key);
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
    }
}

