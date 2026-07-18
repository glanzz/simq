//! Checkpointing for resumable execution
//!
//! Checkpoints capture a real, restorable snapshot of the quantum state.
//! `create_checkpoint` serializes the state (sparse-aware, so a 30-qubit
//! state with a handful of amplitudes stays small) and `restore_checkpoint`
//! reconstructs it exactly. An earlier version stored an empty placeholder
//! snapshot and returned `Ok` — a checkpoint that could never be restored;
//! that silent failure mode is gone.

use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use simq_state::{AdaptiveState, DenseState, SparseState};
use std::path::PathBuf;

/// Serialization format version for state snapshots
const SNAPSHOT_VERSION: u8 = 1;

/// Representation tags in the snapshot header
const TAG_SPARSE: u8 = 0;
const TAG_DENSE: u8 = 1;

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

    pub fn create_checkpoint(&mut self, gate_index: usize, state: &AdaptiveState) -> Result<()> {
        let checkpoint = Checkpoint {
            gate_index,
            state_snapshot: serialize_state(state),
            timestamp: std::time::SystemTime::now(),
        };

        self.checkpoints.push(checkpoint);

        if self.checkpoints.len() > self.max_checkpoints {
            self.checkpoints.remove(0);
        }

        Ok(())
    }

    /// Restore the checkpoint at the given index, returning the gate index it
    /// was taken at and the reconstructed state.
    pub fn restore_checkpoint(&self, index: usize) -> Result<(usize, AdaptiveState)> {
        let checkpoint =
            self.checkpoints
                .get(index)
                .ok_or_else(|| ExecutionError::CheckpointFailed {
                    reason: format!(
                        "no checkpoint at index {} ({} stored)",
                        index,
                        self.checkpoints.len()
                    ),
                })?;
        let state = deserialize_state(&checkpoint.state_snapshot)?;
        Ok((checkpoint.gate_index, state))
    }

    pub fn latest_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    /// Number of stored checkpoints
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Whether any checkpoints are stored
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
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

/// Serialize an adaptive state, preserving its representation.
///
/// Layout (little-endian):
/// - `u8` version
/// - `u8` representation tag (0 = sparse, 1 = dense)
/// - `u64` number of qubits
/// - sparse: `u64` entry count, then (`u64` basis index, `f64` re, `f64` im)*
/// - dense: `f64` re, `f64` im for all 2^n amplitudes in order
fn serialize_state(state: &AdaptiveState) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(SNAPSHOT_VERSION);

    match state {
        AdaptiveState::Sparse { state: sparse, .. } => {
            bytes.push(TAG_SPARSE);
            bytes.extend_from_slice(&(sparse.num_qubits() as u64).to_le_bytes());
            // Sort for deterministic snapshots (the map iterates in hash order)
            let mut entries: Vec<(u64, Complex64)> =
                sparse.amplitudes().iter().map(|(&k, &v)| (k, v)).collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
            for (index, amp) in entries {
                bytes.extend_from_slice(&index.to_le_bytes());
                bytes.extend_from_slice(&amp.re.to_le_bytes());
                bytes.extend_from_slice(&amp.im.to_le_bytes());
            }
        },
        AdaptiveState::Dense(dense) => {
            bytes.push(TAG_DENSE);
            bytes.extend_from_slice(&(dense.num_qubits() as u64).to_le_bytes());
            for amp in dense.amplitudes() {
                bytes.extend_from_slice(&amp.re.to_le_bytes());
                bytes.extend_from_slice(&amp.im.to_le_bytes());
            }
        },
    }

    bytes
}

/// Reconstruct a state from a snapshot produced by [`serialize_state`]
fn deserialize_state(bytes: &[u8]) -> Result<AdaptiveState> {
    let fail = |reason: String| ExecutionError::CheckpointFailed { reason };

    let mut cursor = Cursor { bytes, pos: 0 };
    let version = cursor.take_u8()?;
    if version != SNAPSHOT_VERSION {
        return Err(fail(format!(
            "unsupported snapshot version {} (expected {})",
            version, SNAPSHOT_VERSION
        )));
    }
    let tag = cursor.take_u8()?;
    let num_qubits = cursor.take_u64()? as usize;
    if num_qubits > 63 {
        return Err(fail(format!("corrupt snapshot: {} qubits", num_qubits)));
    }

    match tag {
        TAG_SPARSE => {
            let count = cursor.take_u64()? as usize;
            let mut sparse = SparseState::new(num_qubits).map_err(ExecutionError::StateError)?;
            // Clear the default |0...0⟩ amplitude before loading the snapshot
            sparse.amplitudes_mut().clear();
            let dimension = 1u64 << num_qubits;
            for _ in 0..count {
                let index = cursor.take_u64()?;
                if index >= dimension {
                    return Err(fail(format!(
                        "corrupt snapshot: basis index {} out of range for {} qubits",
                        index, num_qubits
                    )));
                }
                let re = cursor.take_f64()?;
                let im = cursor.take_f64()?;
                sparse.set_amplitude(index, Complex64::new(re, im));
            }
            sparse.update_density();
            cursor.expect_end()?;
            Ok(AdaptiveState::Sparse {
                state: sparse,
                threshold: 0.1,
            })
        },
        TAG_DENSE => {
            let dimension = 1usize << num_qubits;
            let mut amplitudes = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                let re = cursor.take_f64()?;
                let im = cursor.take_f64()?;
                amplitudes.push(Complex64::new(re, im));
            }
            cursor.expect_end()?;
            let dense = DenseState::from_amplitudes(num_qubits, &amplitudes)
                .map_err(ExecutionError::StateError)?;
            Ok(AdaptiveState::Dense(dense))
        },
        other => Err(fail(format!("unknown representation tag {}", other))),
    }
}

/// Minimal byte-slice reader with bounds-checked reads
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8]> {
        let end = self.pos.checked_add(len).filter(|&e| e <= self.bytes.len());
        match end {
            Some(end) => {
                let slice = &self.bytes[self.pos..end];
                self.pos = end;
                Ok(slice)
            },
            None => Err(ExecutionError::CheckpointFailed {
                reason: "corrupt snapshot: truncated data".to_string(),
            }),
        }
    }

    fn take_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn take_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("length 8")))
    }

    fn take_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().expect("length 8")))
    }

    fn expect_end(&self) -> Result<()> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(ExecutionError::CheckpointFailed {
                reason: format!("corrupt snapshot: {} trailing bytes", self.bytes.len() - self.pos),
            })
        }
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
        assert!(manager.is_empty());
    }

    #[test]
    fn test_create_and_retrieve_checkpoint() {
        let mut manager = CheckpointManager::new(5);
        let state = AdaptiveState::new(2).unwrap();
        manager.create_checkpoint(0, &state).unwrap();
        assert!(manager.latest_checkpoint().is_some());
        assert_eq!(manager.latest_checkpoint().unwrap().gate_index, 0);
        // Snapshots must not be empty placeholders
        assert!(!manager
            .latest_checkpoint()
            .unwrap()
            .state_snapshot
            .is_empty());
    }

    #[test]
    fn test_restore_missing_checkpoint_errors() {
        let manager = CheckpointManager::new(5);
        let result = manager.restore_checkpoint(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_round_trip() {
        let mut manager = CheckpointManager::new(5);
        // 5 qubits: a single-amplitude state has density 1/32, well under the
        // 10% sparse→dense threshold, so it stays sparse after the gate.
        let mut state = AdaptiveState::new(5).unwrap();
        // Put the state into a nontrivial sparse configuration: X on qubit 1
        let x = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];
        state.apply_single_qubit_gate(&x, 1).unwrap();
        assert!(state.is_sparse());

        manager.create_checkpoint(7, &state).unwrap();
        let (gate_index, restored) = manager.restore_checkpoint(0).unwrap();
        assert_eq!(gate_index, 7);
        assert!(restored.is_sparse(), "sparse states must restore as sparse");

        let original = state.to_dense_vec();
        let round_trip = restored.to_dense_vec();
        assert_eq!(original.len(), round_trip.len());
        for (a, b) in original.iter().zip(&round_trip) {
            assert!((a - b).norm() < 1e-15, "amplitudes differ: {:?} vs {:?}", a, b);
        }
    }

    #[test]
    fn test_dense_round_trip() {
        let mut manager = CheckpointManager::new(5);
        let mut state = AdaptiveState::new(2).unwrap();
        state.force_to_dense().unwrap();
        // Superpose with complex phases: H then S-like phase via raw gate
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let hadamard = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        let s = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
        ];
        state.apply_single_qubit_gate(&hadamard, 0).unwrap();
        state.apply_single_qubit_gate(&s, 0).unwrap();
        assert!(state.is_dense());

        manager.create_checkpoint(3, &state).unwrap();
        let (gate_index, restored) = manager.restore_checkpoint(0).unwrap();
        assert_eq!(gate_index, 3);
        assert!(restored.is_dense(), "dense states must restore as dense");

        let original = state.to_dense_vec();
        let round_trip = restored.to_dense_vec();
        for (a, b) in original.iter().zip(&round_trip) {
            assert!((a - b).norm() < 1e-15);
        }
    }

    #[test]
    fn test_corrupt_snapshot_rejected() {
        // Truncated snapshot
        assert!(deserialize_state(&[SNAPSHOT_VERSION, TAG_DENSE, 3]).is_err());
        // Unknown version
        assert!(deserialize_state(&[42, TAG_DENSE]).is_err());
        // Unknown tag
        let mut bytes = vec![SNAPSHOT_VERSION, 9];
        bytes.extend_from_slice(&1u64.to_le_bytes());
        assert!(deserialize_state(&bytes).is_err());
    }

    #[test]
    fn test_corrupt_snapshot_too_many_qubits_rejected() {
        // num_qubits > 63 must be rejected before any allocation is attempted
        let mut bytes = vec![SNAPSHOT_VERSION, TAG_DENSE];
        bytes.extend_from_slice(&64u64.to_le_bytes());
        let err = deserialize_state(&bytes).unwrap_err();
        assert!(format!("{}", err).contains("64") || format!("{:?}", err).contains("64"));
    }

    #[test]
    fn test_corrupt_snapshot_sparse_index_out_of_range_rejected() {
        // Sparse snapshot for 2 qubits (dimension 4) with a basis index of 10,
        // which is out of range and must be rejected.
        let mut bytes = vec![SNAPSHOT_VERSION, TAG_SPARSE];
        bytes.extend_from_slice(&2u64.to_le_bytes()); // num_qubits
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one entry
        bytes.extend_from_slice(&10u64.to_le_bytes()); // out-of-range index
        bytes.extend_from_slice(&1.0f64.to_le_bytes()); // re
        bytes.extend_from_slice(&0.0f64.to_le_bytes()); // im
        let err = deserialize_state(&bytes).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("out of range"), "unexpected error: {}", msg);
    }

    #[test]
    fn test_corrupt_snapshot_trailing_bytes_rejected() {
        // Well-formed dense snapshot for 1 qubit, plus extra trailing garbage.
        let mut bytes = vec![SNAPSHOT_VERSION, TAG_DENSE];
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.push(0xFF); // trailing garbage byte
        let err = deserialize_state(&bytes).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("trailing"), "unexpected error: {}", msg);
    }

    #[test]
    fn test_checkpoint_eviction() {
        let mut manager = CheckpointManager::new(2);
        let state = AdaptiveState::new(1).unwrap();
        manager.create_checkpoint(0, &state).unwrap();
        manager.create_checkpoint(1, &state).unwrap();
        manager.create_checkpoint(2, &state).unwrap(); // should evict oldest
        assert_eq!(manager.len(), 2);
        // The oldest checkpoint (gate 0) was evicted; index 0 is now gate 1
        assert_eq!(manager.restore_checkpoint(0).unwrap().0, 1);
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
