use simq_state::DenseState;
use std::time::Instant;

fn main() {
    println!("SimQ Memory Usage Benchmark");
    println!("===========================");

    let sizes = [10, 15, 20, 22, 24, 25]; // 25 qubits = 32M complex numbers = 512MB

    println!("{:<10} | {:<15} | {:<15} | {:<15}", "Qubits", "State Size", "Theoretical", "Allocation Time");
    println!("{:-<10}-+-{:-<15}-+-{:-<15}-+-{:-<15}", "", "", "", "");

    for &n in &sizes {
        let dim = 1 << n;
        let theoretical_bytes = dim * 16; // 128-bit complex numbers (2 * f64)
        
        let start = Instant::now();
        let state = DenseState::new(n);
        let duration = start.elapsed();

        match state {
            Ok(s) => {
                let size_str = format_size(theoretical_bytes);
                println!("{:<10} | {:<15} | {:<15} | {:?}", n, dim, size_str, duration);
                
                // Keep state alive to prevent optimization
                if s.num_qubits() != n {
                    println!("Error: size mismatch");
                }
            },
            Err(e) => {
                println!("{:<10} | {:<15} | Failed: {}", n, dim, e);
            }
        }
    }
}

fn format_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
