//! Automatic differentiation for variational quantum circuits in SimQ

use std::cell::RefCell;
use std::rc::Rc;

/// Differentiable parameter for AD
#[derive(Clone)]
pub struct DifferentiableParameter {
    pub value: f64,
    pub grad: Rc<RefCell<f64>>,
}

impl DifferentiableParameter {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            grad: Rc::new(RefCell::new(0.0)),
        }
    }
    pub fn set_grad(&self, g: f64) {
        *self.grad.borrow_mut() = g;
    }
    pub fn get_grad(&self) -> f64 {
        *self.grad.borrow()
    }
}

/// Example: Compute gradients for expectation value via reverse-mode AD
/// This is a stub illustrating the API; actual implementation would require circuit graph traversal.
pub fn compute_gradients_ad(params: &[DifferentiableParameter], expectation_fn: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    // Forward pass: compute output
    let param_values: Vec<f64> = params.iter().map(|p| p.value).collect();
    let output = expectation_fn(&param_values);
    // Reverse pass: finite difference for illustration (replace with real AD graph)
    let eps = 1e-8;
    let mut grads = Vec::with_capacity(params.len());
    for (i, p) in params.iter().enumerate() {
        let mut shifted = param_values.clone();
        shifted[i] += eps;
        let out_shifted = expectation_fn(&shifted);
        let grad = (out_shifted - output) / eps;
        p.set_grad(grad);
        grads.push(grad);
    }
    grads
}
