| Workload | SimQ | Qiskit Statevector | Qiskit Aer | vs Statevector | vs Aer |
|---|---|---|---|---|---|
| `vqe_energy/4q` | 0.047 ms | 1.770 ms | 1.801 ms | **38.1× faster** | **38.7× faster** |
| `vqe_energy/8q` | 0.652 ms | 3.867 ms | 2.924 ms | **5.9× faster** | **4.5× faster** |
| `vqe_energy/12q` | 14.700 ms | 7.622 ms | 5.652 ms | 1.9× slower | 2.6× slower |
| `vqe_energy/16q` | 385.322 ms | 120.392 ms | 17.611 ms | 3.2× slower | 21.9× slower |
| `qaoa_maxcut/4q` | 0.033 ms | 1.127 ms | 1.462 ms | **34.1× faster** | **44.3× faster** |
| `qaoa_maxcut/8q` | 0.511 ms | 2.361 ms | 2.221 ms | **4.6× faster** | **4.3× faster** |
| `qaoa_maxcut/12q` | 11.649 ms | 4.628 ms | 4.131 ms | 2.5× slower | 2.8× slower |
| `qaoa_maxcut/16q` | 310.474 ms | 72.244 ms | 10.214 ms | 4.3× slower | 30.4× slower |
| `ghz_sampling/10q` | 0.045 ms | 4.926 ms | 2.097 ms | **109.0× faster** | **46.4× faster** |
| `ghz_sampling/16q` | 0.671 ms | 634.268 ms | 4.497 ms | **944.7× faster** | **6.7× faster** |
| `ghz_sampling/20q` | 42.692 ms | 15.88 s | 36.562 ms | **372.0× faster** | 1.2× slower |
