//! State validation and verification

use crate::execution_engine::error::{ExecutionError, Result};
use simq_state::AdaptiveState;

/// Validate that a quantum state is normalized
pub fn validate_normalization(state: &AdaptiveState, tolerance: f64) -> Result<()> {
    let norm_sq = match state {
        AdaptiveState::Dense(dense) => dense.amplitudes().iter().map(|a| a.norm_sqr()).sum::<f64>(),
        AdaptiveState::Sparse { state: sparse, .. } => sparse
            .amplitudes()
            .values()
            .map(|a| a.norm_sqr())
            .sum::<f64>(),
    };

    let norm = norm_sq.sqrt();

    if (norm - 1.0).abs() > tolerance {
        return Err(ExecutionError::ValidationFailed {
            reason: format!("State not normalized: norm = {} (expected 1.0)", norm),
        });
    }

    Ok(())
}

/// Validate that amplitudes don't contain NaN or Inf
pub fn validate_finite(state: &AdaptiveState) -> Result<()> {
    let has_invalid = match state {
        AdaptiveState::Dense(dense) => dense
            .amplitudes()
            .iter()
            .any(|a| !a.re.is_finite() || !a.im.is_finite()),
        AdaptiveState::Sparse { state: sparse, .. } => sparse
            .amplitudes()
            .values()
            .any(|a| !a.re.is_finite() || !a.im.is_finite()),
    };

    if has_invalid {
        return Err(ExecutionError::ValidationFailed {
            reason: "State contains NaN or Inf values".to_string(),
        });
    }

    Ok(())
}

/// Full state validation
pub fn validate_state(state: &AdaptiveState) -> Result<()> {
    validate_finite(state)?;
    validate_normalization(state, 1e-6)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_validation() {
        let state = simq_state::DenseState::new(1).unwrap();
        let adaptive = AdaptiveState::Dense(state);
        assert!(validate_normalization(&adaptive, 1e-6).is_ok());
    }
}
