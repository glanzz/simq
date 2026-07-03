//! Checkpointing for resumable execution

use crate::execution_engine::error::Result;
use simq_state::AdaptiveState;
use std::path::PathBuf;

/// A checkpoint of execution state
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub gate_index: usize,
    pub state_snapshot: Vec<u8>, // Serialized state
    pub timestamp: std::time::SystemTime,
}

/// Manages checkpoints during execution
#[derive(Debug)]
pub struct CheckpointManager {
    checkpoints: Vec<Checkpoint>,
    max_checkpoints: usize,
    checkpoint_dir: Option<PathBuf>,
}

impl CheckpointManager {
    pub fn new(max_checkpoints: usize) -> Self {
        Self {
            checkpoints: Vec::new(),
            max_checkpoints,
            checkpoint_dir: None,
        }
    }

    pub fn with_directory(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    pub fn create_checkpoint(&mut self, gate_index: usize, _state: &AdaptiveState) -> Result<()> {
        // TODO: Implement state serialization
        let checkpoint = Checkpoint {
            gate_index,
            state_snapshot: vec![], // Placeholder
            timestamp: std::time::SystemTime::now(),
        };

        self.checkpoints.push(checkpoint);

        if self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }

        Ok(())
    }

    pub fn restore_checkpoint(&self, _index: usize) -> Result<(usize, AdaptiveState)> {
        // TODO: Implement state deserialization
        Err(crate::execution_engine::error::ExecutionError::CheckpointFailed {
            reason: "Not implemented".to_string(),
        })
    }

    pub fn latest_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self::new(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_state::AdaptiveState;

    #[test]
    fn test_default_checkpoint_manager() {
        let manager = CheckpointManager::default();
        assert_eq!(manager.max_checkpoints, 10);
        assert!(manager.latest_checkpoint().is_none());
    }

    #[test]
    fn test_create_and_retrieve_checkpoint() {
        let mut manager = CheckpointManager::new(5);
        let state = AdaptiveState::new(2).unwrap();
        manager.create_checkpoint(0, &state).unwrap();
        assert!(manager.latest_checkpoint().is_some());
        assert_eq!(manager.latest_checkpoint().unwrap().gate_index, 0);
    }

    #[test]
    fn test_restore_checkpoint_not_implemented() {
        let manager = CheckpointManager::new(5);
        let result = manager.restore_checkpoint(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_eviction() {
        let mut manager = CheckpointManager::new(2);
        let state = AdaptiveState::new(1).unwrap();
        manager.create_checkpoint(0, &state).unwrap();
        manager.create_checkpoint(1, &state).unwrap();
        manager.create_checkpoint(2, &state).unwrap(); // should evict oldest
        assert_eq!(manager.checkpoints.len(), 2);
    }

    #[test]
    fn test_with_directory() {
        use std::path::PathBuf;
        let manager = CheckpointManager::new(5).with_directory(PathBuf::from("/tmp"));
        assert!(manager.checkpoint_dir.is_some());
    }

    #[test]
    fn test_clear() {
        let mut manager = CheckpointManager::new(5);
        let state = AdaptiveState::new(1).unwrap();
        manager.create_checkpoint(0, &state).unwrap();
        manager.clear();
        assert!(manager.latest_checkpoint().is_none());
    }
}
