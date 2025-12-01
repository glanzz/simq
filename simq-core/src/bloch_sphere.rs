//! Bloch sphere visualization for single-qubit states
//!
//! The Bloch sphere is a geometric representation of pure single-qubit quantum states.
//! Any pure single-qubit state can be written as:
//!
//! |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
//!
//! where θ ∈ [0, π] and φ ∈ [0, 2π) define a point on the unit sphere.
//!
//! # Example
//!
//! ```
//! use simq_core::BlochVector;
//! use num_complex::Complex64;
//!
//! // |0⟩ state points to north pole
//! let state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
//! let bloch = BlochVector::from_state(&state);
//! assert!((bloch.z - 1.0).abs() < 1e-10);
//!
//! // |+⟩ state points along +x axis
//! let plus = [
//!     Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
//!     Complex64::new(1.0/2.0_f64.sqrt(), 0.0)
//! ];
//! let bloch = BlochVector::from_state(&plus);
//! assert!((bloch.x - 1.0).abs() < 1e-10);
//! ```

use num_complex::Complex64;
use std::f64::consts::PI;
use std::fmt;

/// A point on the Bloch sphere represented in Cartesian coordinates
#[derive(Clone, Debug)]
pub struct BlochVector {
    /// X coordinate (-1 to 1)
    pub x: f64,
    /// Y coordinate (-1 to 1)
    pub y: f64,
    /// Z coordinate (-1 to 1), where +Z is |0⟩ and -Z is |1⟩
    pub z: f64,
}

/// Bloch sphere angles (spherical coordinates)
#[derive(Clone, Debug)]
pub struct BlochAngles {
    /// Polar angle θ ∈ [0, π]
    pub theta: f64,
    /// Azimuthal angle φ ∈ [0, 2π)
    pub phi: f64,
}

impl BlochVector {
    /// Create a Bloch vector from Cartesian coordinates
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Convert a single-qubit state to a Bloch vector
    ///
    /// # Arguments
    /// * `state` - Two-element array representing [α, β] where |ψ⟩ = α|0⟩ + β|1⟩
    ///
    /// # Returns
    /// Bloch vector representation
    ///
    /// # Example
    /// ```
    /// use simq_core::BlochVector;
    /// use num_complex::Complex64;
    ///
    /// let state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    /// let bloch = BlochVector::from_state(&state);
    /// ```
    pub fn from_state(state: &[Complex64; 2]) -> Self {
        let alpha = state[0];
        let beta = state[1];

        // Bloch vector components using Pauli expectation values
        // x = ⟨σ_x⟩ = 2Re(α*β*)
        // y = ⟨σ_y⟩ = 2Im(α*β*)
        // z = ⟨σ_z⟩ = |α|² - |β|²

        let alpha_conj_beta = alpha.conj() * beta;

        let x = 2.0 * alpha_conj_beta.re;
        let y = 2.0 * alpha_conj_beta.im;
        let z = alpha.norm_sqr() - beta.norm_sqr();

        Self { x, y, z }
    }

    /// Convert Bloch vector to spherical coordinates
    pub fn to_angles(&self) -> BlochAngles {
        let r = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();

        // Handle near-zero vector
        if r < 1e-10 {
            return BlochAngles {
                theta: 0.0,
                phi: 0.0,
            };
        }

        // θ = arccos(z/r)
        let theta = (self.z / r).acos();

        // φ = atan2(y, x)
        let phi = self.y.atan2(self.x);
        let phi = if phi < 0.0 { phi + 2.0 * PI } else { phi };

        BlochAngles { theta, phi }
    }

    /// Get the magnitude of the Bloch vector
    ///
    /// For pure states, this should be 1.0
    /// For mixed states, this is less than 1.0
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Check if this represents a pure state (magnitude ≈ 1.0)
    pub fn is_pure(&self, tolerance: f64) -> bool {
        (self.magnitude() - 1.0).abs() < tolerance
    }

    /// Render the Bloch sphere as ASCII art
    ///
    /// Returns a multi-line string showing a 2D projection of the Bloch sphere
    /// with the state vector indicated.
    pub fn render_ascii(&self) -> String {
        BlochRenderer::new().render(self)
    }

    /// Render with custom configuration
    pub fn render_ascii_with_config(&self, config: &BlochRenderConfig) -> String {
        BlochRenderer::with_config(config.clone()).render(self)
    }

    /// Get a description of the state
    pub fn describe(&self) -> String {
        let mag = self.magnitude();
        let angles = self.to_angles();

        let mut desc = String::new();
        desc.push_str(&format!("Bloch Vector: ({:.4}, {:.4}, {:.4})\n", self.x, self.y, self.z));
        desc.push_str(&format!("Magnitude: {:.4}\n", mag));
        desc.push_str(&format!("Angles: θ={:.4}, φ={:.4}\n", angles.theta, angles.phi));

        // Identify special states
        if (self.z - 1.0).abs() < 0.01 {
            desc.push_str("State: |0⟩ (north pole)\n");
        } else if (self.z + 1.0).abs() < 0.01 {
            desc.push_str("State: |1⟩ (south pole)\n");
        } else if (self.x - 1.0).abs() < 0.01 && self.z.abs() < 0.01 {
            desc.push_str("State: |+⟩ (east pole)\n");
        } else if (self.x + 1.0).abs() < 0.01 && self.z.abs() < 0.01 {
            desc.push_str("State: |−⟩ (west pole)\n");
        } else if (self.y - 1.0).abs() < 0.01 && self.z.abs() < 0.01 {
            desc.push_str("State: |+i⟩ (front pole)\n");
        } else if (self.y + 1.0).abs() < 0.01 && self.z.abs() < 0.01 {
            desc.push_str("State: |−i⟩ (back pole)\n");
        } else if mag < 0.9 {
            desc.push_str("Mixed state (inside sphere)\n");
        }

        desc
    }
}

/// Configuration for Bloch sphere rendering
#[derive(Clone, Debug)]
pub struct BlochRenderConfig {
    /// Size of the sphere (radius in characters)
    pub size: usize,
    /// Show axis labels
    pub show_labels: bool,
    /// Show grid lines
    pub show_grid: bool,
    /// Character for the state point
    pub point_char: char,
    /// Show numerical coordinates
    pub show_coords: bool,
}

impl Default for BlochRenderConfig {
    fn default() -> Self {
        Self {
            size: 12,
            show_labels: true,
            show_grid: true,
            point_char: '●',
            show_coords: true,
        }
    }
}

/// Bloch sphere renderer
struct BlochRenderer {
    config: BlochRenderConfig,
}

impl BlochRenderer {
    fn new() -> Self {
        Self {
            config: BlochRenderConfig::default(),
        }
    }

    fn with_config(config: BlochRenderConfig) -> Self {
        Self { config }
    }

    /// Render the Bloch sphere with the state vector
    fn render(&self, vector: &BlochVector) -> String {
        let mut output = String::new();

        // Header
        output.push_str("Bloch Sphere Visualization\n");
        if self.config.show_coords {
            output
                .push_str(&format!("State: ({:.3}, {:.3}, {:.3})\n", vector.x, vector.y, vector.z));
        }
        output.push('\n');

        // XZ plane projection (side view)
        output.push_str("Side View (XZ plane):\n");
        output.push_str(&self.render_xz_plane(vector));
        output.push('\n');

        // XY plane projection (top view)
        output.push_str("Top View (XY plane):\n");
        output.push_str(&self.render_xy_plane(vector));

        output
    }

    /// Render XZ plane (shows |0⟩ at top, |1⟩ at bottom, |+⟩ and |−⟩ on sides)
    fn render_xz_plane(&self, vector: &BlochVector) -> String {
        let size = self.config.size as i32;
        let mut grid = vec![vec![' '; (size * 2 + 1) as usize]; (size + 1) as usize];

        // Draw circle
        for angle in 0..360 {
            let rad = (angle as f64).to_radians();
            let x = (rad.cos() * size as f64) as i32;
            let z = (rad.sin() * size as f64) as i32;

            let gx = (size + x) as usize;
            let gz = (size / 2 - z / 2) as usize;

            if gz < grid.len() && gx < grid[0].len() {
                grid[gz][gx] = '·';
            }
        }

        // Draw axes
        for i in 0..grid.len() {
            let mid = size as usize;
            if i < grid.len() && mid < grid[0].len() {
                grid[i][mid] = '│'; // Z axis
            }
        }
        for j in 0..grid[0].len() {
            let mid = size as usize / 2;
            if mid < grid.len() {
                grid[mid][j] = '─'; // X axis
            }
        }

        // Mark center
        let center_y = size as usize / 2;
        let center_x = size as usize;
        if center_y < grid.len() && center_x < grid[0].len() {
            grid[center_y][center_x] = '┼';
        }

        // Plot state vector
        let state_x = (vector.x * size as f64) as i32;
        let state_z = (vector.z * size as f64) as i32;
        let gx = (size + state_x) as usize;
        let gz = (size / 2 - state_z / 2) as usize;

        if gz < grid.len() && gx < grid[0].len() {
            grid[gz][gx] = self.config.point_char;
        }

        // Add labels
        let mut result = String::new();
        if self.config.show_labels {
            result.push_str(&format!("{:>width$}|0⟩\n", "", width = size as usize));
        }

        for row in &grid {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        if self.config.show_labels {
            result.push_str(&format!("{:>width$}|1⟩\n", "", width = size as usize));
            result.push_str(&format!("|−⟩{:─<width$}|+⟩\n", "", width = (size * 2 - 4) as usize));
        }

        result
    }

    /// Render XY plane (shows |+⟩, |−⟩, |+i⟩, |−i⟩)
    fn render_xy_plane(&self, vector: &BlochVector) -> String {
        let size = self.config.size as i32;
        let mut grid = vec![vec![' '; (size * 2 + 1) as usize]; (size + 1) as usize];

        // Draw circle
        for angle in 0..360 {
            let rad = (angle as f64).to_radians();
            let x = (rad.cos() * size as f64) as i32;
            let y = (rad.sin() * size as f64) as i32;

            let gx = (size + x) as usize;
            let gy = (size / 2 - y / 2) as usize;

            if gy < grid.len() && gx < grid[0].len() {
                grid[gy][gx] = '·';
            }
        }

        // Draw axes
        for i in 0..grid.len() {
            let mid = size as usize;
            if mid < grid[0].len() {
                grid[i][mid] = '│'; // Y axis
            }
        }
        for j in 0..grid[0].len() {
            let mid = size as usize / 2;
            if mid < grid.len() {
                grid[mid][j] = '─'; // X axis
            }
        }

        // Mark center
        let center_y = size as usize / 2;
        let center_x = size as usize;
        if center_y < grid.len() && center_x < grid[0].len() {
            grid[center_y][center_x] = '┼';
        }

        // Plot state vector (projected onto XY plane)
        let state_x = (vector.x * size as f64) as i32;
        let state_y = (vector.y * size as f64) as i32;
        let gx = (size + state_x) as usize;
        let gy = (size / 2 - state_y / 2) as usize;

        if gy < grid.len() && gx < grid[0].len() {
            grid[gy][gx] = self.config.point_char;
        }

        // Convert to string
        let mut result = String::new();
        if self.config.show_labels {
            result.push_str(&format!("{:>width$}|+i⟩\n", "", width = size as usize));
        }

        for row in &grid {
            result.push_str(&row.iter().collect::<String>());
            result.push('\n');
        }

        if self.config.show_labels {
            result.push_str(&format!("{:>width$}|−i⟩\n", "", width = size as usize));
            result.push_str(&format!("|−⟩{:─<width$}|+⟩\n", "", width = (size * 2 - 4) as usize));
        }

        result
    }
}

impl fmt::Display for BlochVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BlochVector({:.4}, {:.4}, {:.4})", self.x, self.y, self.z)
    }
}

impl BlochAngles {
    /// Convert spherical coordinates to a Bloch vector
    pub fn to_vector(&self) -> BlochVector {
        let x = self.theta.sin() * self.phi.cos();
        let y = self.theta.sin() * self.phi.sin();
        let z = self.theta.cos();
        BlochVector { x, y, z }
    }

    /// Convert to quantum state coefficients
    ///
    /// Returns [α, β] where |ψ⟩ = α|0⟩ + β|1⟩
    pub fn to_state(&self) -> [Complex64; 2] {
        let alpha = Complex64::new((self.theta / 2.0).cos(), 0.0);
        let beta = Complex64::new(
            (self.theta / 2.0).sin() * self.phi.cos(),
            (self.theta / 2.0).sin() * self.phi.sin(),
        );
        [alpha, beta]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state() {
        let state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let bloch = BlochVector::from_state(&state);

        assert!((bloch.x).abs() < 1e-10);
        assert!((bloch.y).abs() < 1e-10);
        assert!((bloch.z - 1.0).abs() < 1e-10);
        assert!(bloch.is_pure(1e-10));
    }

    #[test]
    fn test_one_state() {
        let state = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let bloch = BlochVector::from_state(&state);

        assert!((bloch.x).abs() < 1e-10);
        assert!((bloch.y).abs() < 1e-10);
        assert!((bloch.z + 1.0).abs() < 1e-10);
        assert!(bloch.is_pure(1e-10));
    }

    #[test]
    fn test_plus_state() {
        let state = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let bloch = BlochVector::from_state(&state);

        assert!((bloch.x - 1.0).abs() < 1e-10);
        assert!((bloch.y).abs() < 1e-10);
        assert!((bloch.z).abs() < 1e-10);
        assert!(bloch.is_pure(1e-10));
    }

    #[test]
    fn test_minus_state() {
        let state = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let bloch = BlochVector::from_state(&state);

        assert!((bloch.x + 1.0).abs() < 1e-10);
        assert!((bloch.y).abs() < 1e-10);
        assert!((bloch.z).abs() < 1e-10);
        assert!(bloch.is_pure(1e-10));
    }

    #[test]
    fn test_plus_i_state() {
        let state = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 1.0 / 2.0_f64.sqrt()),
        ];
        let bloch = BlochVector::from_state(&state);

        assert!((bloch.x).abs() < 1e-10);
        assert!((bloch.y - 1.0).abs() < 1e-10);
        assert!((bloch.z).abs() < 1e-10);
        assert!(bloch.is_pure(1e-10));
    }

    #[test]
    fn test_angles_conversion() {
        let angles = BlochAngles {
            theta: PI / 2.0,
            phi: 0.0,
        };
        let vector = angles.to_vector();

        assert!((vector.x - 1.0).abs() < 1e-10);
        assert!((vector.y).abs() < 1e-10);
        assert!((vector.z).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_conversion() {
        // Use a normalized vector (on unit sphere)
        let original = BlochVector::new(0.0, 0.707, 0.707);
        let angles = original.to_angles();
        let reconstructed = angles.to_vector();

        assert!((original.x - reconstructed.x).abs() < 1e-2);
        assert!((original.y - reconstructed.y).abs() < 1e-2);
        assert!((original.z - reconstructed.z).abs() < 1e-2);
    }

    #[test]
    fn test_render_ascii() {
        let state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let bloch = BlochVector::from_state(&state);

        let output = bloch.render_ascii();
        assert!(output.contains("Bloch Sphere"));
        assert!(output.contains("|0⟩"));
    }
}
