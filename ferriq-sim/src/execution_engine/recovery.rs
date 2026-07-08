//! Error recovery policies

/// Policy for error recovery during execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecoveryPolicy {
    /// Stop execution on first error
    #[default]
    Halt,
    /// Skip failed gate and continue
    Skip,
    /// Attempt to retry failed gate once
    RetryOnce,
    /// Retry with exponential backoff
    RetryWithBackoff { max_attempts: usize },
    /// Attempt fallback strategy (e.g., CPU if GPU fails)
    Fallback,
}

impl RecoveryPolicy {
    pub fn should_retry(&self, attempt: usize) -> bool {
        match self {
            Self::Halt => false,
            Self::Skip => false,
            Self::RetryOnce => attempt < 2,
            Self::RetryWithBackoff { max_attempts } => attempt < *max_attempts,
            Self::Fallback => attempt < 2,
        }
    }

    pub fn max_attempts(&self) -> usize {
        match self {
            Self::Halt => 1,
            Self::Skip => 1,
            Self::RetryOnce => 2,
            Self::RetryWithBackoff { max_attempts } => *max_attempts,
            Self::Fallback => 2,
        }
    }
}
