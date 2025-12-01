//! Advanced batch evaluation features
//!
//! This module extends basic batch evaluation with advanced features for
//! production use cases including adaptive batching, distributed evaluation,
//! and optimization-specific batch strategies.

use super::batch::BatchResult;
use crate::error::Result;
use crate::Simulator;
use rand::distributions::Distribution;
use rayon::prelude::*;
use simq_core::Circuit;
use simq_state::observable::PauliObservable;
use simq_state::AdaptiveState;
use std::time::{Duration, Instant};

/// Configuration for advanced batch evaluation
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (0 = unlimited)
    pub max_batch_size: usize,
    /// Enable adaptive batch sizing
    pub adaptive_sizing: bool,
    /// Timeout per batch (None = no timeout)
    pub timeout: Option<Duration>,
    /// Enable result caching
    pub enable_cache: bool,
    /// Progress callback frequency (0 = disabled)
    pub progress_frequency: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            adaptive_sizing: true,
            timeout: None,
            enable_cache: false,
            progress_frequency: 100,
        }
    }
}

/// Adaptive batch evaluator that automatically tunes batch size
pub struct AdaptiveBatchEvaluator {
    config: BatchConfig,
    avg_eval_time: Option<Duration>,
    successful_batch_sizes: Vec<usize>,
}

impl AdaptiveBatchEvaluator {
    /// Create a new adaptive batch evaluator
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            avg_eval_time: None,
            successful_batch_sizes: Vec::new(),
        }
    }

    /// Evaluate with adaptive batching
    ///
    /// Automatically adjusts batch size based on evaluation time and available resources.
    pub fn evaluate<F>(
        &mut self,
        simulator: &Simulator,
        circuit_builder: F,
        observable: &PauliObservable,
        param_sets: &[Vec<f64>],
    ) -> Result<BatchResult>
    where
        F: Fn(&[f64]) -> Circuit + Send + Sync,
    {
        let start_time = Instant::now();
        let total_params = param_sets.len();

        if total_params == 0 {
            return Ok(BatchResult {
                values: vec![],
                num_evaluations: 0,
                computation_time: Duration::ZERO,
            });
        }

        // Determine initial batch size
        let batch_size = self.estimate_batch_size(total_params);

        // Process in batches
        let mut all_values = Vec::with_capacity(total_params);
        let mut processed = 0;

        for chunk in param_sets.chunks(batch_size) {
            let chunk_start = Instant::now();

            // Evaluate this batch
            let chunk_values: Vec<f64> = chunk
                .par_iter()
                .map(|params| {
                    let circuit = circuit_builder(params);
                    evaluate_single(simulator, &circuit, observable)
                })
                .collect::<Result<Vec<f64>>>()?;

            all_values.extend(chunk_values);
            processed += chunk.len();

            // Update timing statistics
            let chunk_time = chunk_start.elapsed();
            self.update_statistics(chunk.len(), chunk_time);

            // Check timeout
            if let Some(timeout) = self.config.timeout {
                if start_time.elapsed() > timeout {
                    return Err(crate::error::SimulatorError::Other(format!(
                        "Batch evaluation timeout after {} evaluations",
                        processed
                    )));
                }
            }

            // Progress callback
            if self.config.progress_frequency > 0 && processed % self.config.progress_frequency == 0
            {
                eprintln!("Progress: {}/{} evaluations", processed, total_params);
            }
        }

        Ok(BatchResult {
            values: all_values,
            num_evaluations: total_params,
            computation_time: start_time.elapsed(),
        })
    }

    /// Estimate optimal batch size
    fn estimate_batch_size(&self, total: usize) -> usize {
        if !self.config.adaptive_sizing {
            return self.config.max_batch_size.min(total);
        }

        // Use historical data to estimate optimal size
        if let Some(avg_time) = self.avg_eval_time {
            // Aim for batches that take ~1-2 seconds
            let target_time = Duration::from_secs(1);
            let estimated_size = (target_time.as_secs_f64() / avg_time.as_secs_f64()) as usize;

            estimated_size
                .max(10) // At least 10
                .min(self.config.max_batch_size) // At most max
                .min(total) // At most total
        } else {
            // Start conservative
            100.min(self.config.max_batch_size).min(total)
        }
    }

    /// Update timing statistics
    fn update_statistics(&mut self, batch_size: usize, time: Duration) {
        let per_eval = time / batch_size as u32;

        self.avg_eval_time = Some(match self.avg_eval_time {
            None => per_eval,
            Some(prev) => {
                // Exponential moving average
                Duration::from_secs_f64(0.7 * prev.as_secs_f64() + 0.3 * per_eval.as_secs_f64())
            },
        });

        self.successful_batch_sizes.push(batch_size);

        // Keep only recent history
        if self.successful_batch_sizes.len() > 10 {
            self.successful_batch_sizes.remove(0);
        }
    }
}

/// Evaluate single circuit expectation
fn evaluate_single(
    simulator: &Simulator,
    circuit: &Circuit,
    observable: &PauliObservable,
) -> Result<f64> {
    let result = simulator.run(circuit)?;

    let expectation = match &result.state {
        AdaptiveState::Dense(dense) => observable.expectation_value(dense)?,
        AdaptiveState::Sparse { state: sparse, .. } => {
            use simq_state::DenseState;
            let dense = DenseState::from_sparse(sparse)?;
            observable.expectation_value(&dense)?
        },
    };

    Ok(expectation)
}

/// Latin Hypercube Sampling for parameter space exploration
///
/// More efficient than grid search for high-dimensional spaces.
pub fn latin_hypercube_sampling(param_ranges: &[(f64, f64)], num_samples: usize) -> Vec<Vec<f64>> {
    use rand::seq::SliceRandom;
    use rand::Rng;

    let num_params = param_ranges.len();
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(num_samples);

    // Create permutations for each dimension
    let permutations: Vec<Vec<usize>> = (0..num_params)
        .map(|_| {
            let mut perm: Vec<usize> = (0..num_samples).collect();
            perm.shuffle(&mut rng);
            perm
        })
        .collect();

    // Generate samples
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_samples {
        let mut sample = Vec::with_capacity(num_params);

        for (dim, &(min, max)) in param_ranges.iter().enumerate() {
            // Get the permuted index for this dimension
            let bin = permutations[dim][i];

            // Random point within the bin
            let bin_start = bin as f64 / num_samples as f64;
            let bin_end = (bin + 1) as f64 / num_samples as f64;
            let random_in_bin = rng.gen_range(bin_start..bin_end);

            // Scale to actual range
            let value = min + (max - min) * random_in_bin;
            sample.push(value);
        }

        samples.push(sample);
    }

    samples
}

/// Importance sampling for focusing on promising regions
pub struct ImportanceSampler {
    /// Center points of important regions
    centers: Vec<Vec<f64>>,
    /// Weights for each region
    weights: Vec<f64>,
    /// Sampling radius around centers
    radius: f64,
}

impl ImportanceSampler {
    /// Create a new importance sampler
    pub fn new(centers: Vec<Vec<f64>>, weights: Vec<f64>, radius: f64) -> Self {
        assert_eq!(centers.len(), weights.len());
        Self {
            centers,
            weights,
            radius,
        }
    }

    /// Sample parameters from important regions
    pub fn sample(&self, num_samples: usize) -> Vec<Vec<f64>> {
        use rand::distributions::WeightedIndex;
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let dist = WeightedIndex::new(&self.weights).unwrap();

        (0..num_samples)
            .map(|_| {
                // Select a center according to weights
                let center_idx = dist.sample(&mut rng);
                let center = &self.centers[center_idx];

                // Sample near this center
                center
                    .iter()
                    .map(|&c| {
                        let offset = rng.gen_range(-self.radius..self.radius);
                        c + offset
                    })
                    .collect()
            })
            .collect()
    }
}

/// Parallel line search along a direction
///
/// Useful for 1D optimization along a gradient direction.
pub fn line_search<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    start_point: &[f64],
    direction: &[f64],
    step_sizes: &[f64],
) -> Result<(f64, f64)>
// (best_step, best_value)
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    let param_sets: Vec<Vec<f64>> = step_sizes
        .iter()
        .map(|&step| {
            start_point
                .iter()
                .zip(direction.iter())
                .map(|(&p, &d)| p + step * d)
                .collect()
        })
        .collect();

    let batch_result = super::batch::evaluate_batch_expectation(
        simulator,
        circuit_builder,
        observable,
        &param_sets,
    )?;

    let (best_idx, &best_value) = batch_result
        .values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    Ok((step_sizes[best_idx], best_value))
}

/// Batch gradient computation with multiple methods
///
/// Computes gradients using multiple methods simultaneously for verification.
pub fn verify_gradients<F>(
    simulator: &Simulator,
    circuit_builder: F,
    observable: &PauliObservable,
    params: &[f64],
) -> Result<GradientVerification>
where
    F: Fn(&[f64]) -> Circuit + Send + Sync,
{
    use super::finite_difference::{
        compute_gradient_finite_difference, FiniteDifferenceConfig, FiniteDifferenceMethod,
    };
    use super::parameter_shift::{compute_gradient_parameter_shift, ParameterShiftConfig};

    // Compute using both methods in parallel
    let (ps_result, fd_result) = rayon::join(
        || {
            compute_gradient_parameter_shift(
                simulator,
                &circuit_builder,
                observable,
                params,
                &ParameterShiftConfig::default(),
            )
        },
        || {
            compute_gradient_finite_difference(
                simulator,
                &circuit_builder,
                observable,
                params,
                &FiniteDifferenceConfig {
                    method: FiniteDifferenceMethod::Central,
                    epsilon: 1e-7,
                    parallel: true,
                },
            )
        },
    );

    let ps_grad = ps_result?;
    let fd_grad = fd_result?;

    // Compute differences
    let max_diff = ps_grad
        .gradients
        .iter()
        .zip(fd_grad.gradients.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, |a, b| a.max(b));

    let mean_diff = ps_grad
        .gradients
        .iter()
        .zip(fd_grad.gradients.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / ps_grad.gradients.len() as f64;

    Ok(GradientVerification {
        parameter_shift: ps_grad.gradients,
        finite_difference: fd_grad.gradients,
        max_difference: max_diff,
        mean_difference: mean_diff,
        agrees: max_diff < 1e-5, // Threshold for agreement
    })
}

/// Result of gradient verification
#[derive(Debug, Clone)]
pub struct GradientVerification {
    pub parameter_shift: Vec<f64>,
    pub finite_difference: Vec<f64>,
    pub max_difference: f64,
    pub mean_difference: f64,
    pub agrees: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latin_hypercube_sampling() {
        let ranges = vec![(0.0, 1.0), (-1.0, 1.0), (0.0, 2.0)];
        let samples = latin_hypercube_sampling(&ranges, 10);

        assert_eq!(samples.len(), 10);
        assert_eq!(samples[0].len(), 3);

        // Check all samples are within ranges
        for sample in &samples {
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
            assert!(sample[1] >= -1.0 && sample[1] <= 1.0);
            assert!(sample[2] >= 0.0 && sample[2] <= 2.0);
        }
    }

    #[test]
    fn test_importance_sampler() {
        let centers = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let weights = vec![0.7, 0.3];

        let sampler = ImportanceSampler::new(centers, weights, 0.1);
        let samples = sampler.sample(20);

        assert_eq!(samples.len(), 20);
        assert_eq!(samples[0].len(), 2);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 1000);
        assert!(config.adaptive_sizing);
    }
}
