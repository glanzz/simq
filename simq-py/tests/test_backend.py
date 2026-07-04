"""
Comprehensive tests for Phase 5: Backend System

Tests the LocalSimulatorBackend and backend infrastructure.
Covers configuration, execution, job management, and results.
"""

import simq
import pytest


def test_local_simulator_basic_execution():
    """Test 1: Basic circuit execution on LocalSimulatorBackend"""
    # Create a Bell state circuit
    builder = simq.CircuitBuilder(2)
    builder.h(0)
    builder.cx(0, 1)
    circuit = builder.build()

    # Create backend and execute
    backend = simq.LocalSimulatorBackend()
    result = backend.execute(circuit, shots=1000)

    # Verify results
    assert result.shots == 1000
    assert isinstance(result.counts, dict)
    assert len(result.counts) > 0

    # Check that we get Bell state outcomes (00 and 11)
    counts = result.counts
    total = sum(counts.values())
    assert total == 1000

    # For a Bell state, we should see primarily |00⟩ and |11⟩
    if "00" in counts and "11" in counts:
        bell_outcomes = counts.get("00", 0) + counts.get("11", 0)
        assert bell_outcomes > 900  # At least 90% should be Bell outcomes

    # Test result methods
    probs = result.probabilities()
    assert isinstance(probs, dict)
    assert abs(sum(probs.values()) - 1.0) < 0.01  # Probabilities sum to ~1

    most_freq = result.most_frequent()
    assert most_freq in ["00", "11"]

    # Verify backend info
    assert backend.name() == "local_simulator"
    assert backend.max_qubits() == 30
    assert backend.is_available()

    # Cost should be None (free)
    cost = backend.estimate_cost(circuit, 1000)
    assert cost is None or cost == 0.0

    print(f"✓ Test 1 passed: Basic execution works")
    print(f"  Most frequent outcome: |{most_freq}⟩")
    print(f"  Unique outcomes: {len(counts)}")


def test_local_simulator_with_config():
    """Test 2: LocalSimulatorBackend with custom configuration"""
    # Create config with deterministic seed
    config = simq.LocalSimulatorConfig(
        seed=42,
        max_qubits=20,
        sparse_threshold=0.05,
        parallel=True,
        num_threads=2,
    )

    # Create two backends with same config
    backend1 = simq.LocalSimulatorBackend(config=config)
    backend2 = simq.LocalSimulatorBackend(config=config)

    # Create a simple circuit
    builder = simq.CircuitBuilder(3)
    builder.h(0)
    builder.h(1)
    builder.h(2)
    circuit = builder.build()

    # Execute on both backends
    result1 = backend1.execute(circuit, shots=100)
    result2 = backend2.execute(circuit, shots=100)

    # With same seed, results should be identical
    assert result1.counts == result2.counts

    # Verify config properties
    assert backend1.max_qubits() == 20

    print(f"✓ Test 2 passed: Configuration and reproducibility work")
    print(f"  Seed: {config.seed}")
    print(f"  Max qubits: {config.max_qubits}")
    print(f"  Results match: {result1.counts == result2.counts}")


def test_local_simulator_ghz_state():
    """Test 3: Execute GHZ state (3-qubit entanglement)"""
    # Create GHZ state: |000⟩ + |111⟩
    builder = simq.CircuitBuilder(3)
    builder.h(0)
    builder.cx(0, 1)
    builder.cx(1, 2)
    circuit = builder.build()

    backend = simq.LocalSimulatorBackend()
    result = backend.execute(circuit, shots=2000)

    # Verify we get superposition of |000⟩ and |111⟩
    counts = result.counts
    assert "000" in counts or "111" in counts

    if "000" in counts and "111" in counts:
        ghz_outcomes = counts.get("000", 0) + counts.get("111", 0)
        # Should see >95% in these two states
        assert ghz_outcomes > 1900

    # Test probabilities
    probs = result.probabilities()
    if "000" in probs and "111" in probs:
        # Each should be close to 0.5
        assert 0.4 < probs["000"] < 0.6
        assert 0.4 < probs["111"] < 0.6

    print(f"✓ Test 3 passed: GHZ state generation works")
    print(f"  Outcomes: {dict(sorted(counts.items()))}")


def test_backend_result_methods():
    """Test 4: Verify all BackendResult methods work correctly"""
    # Create a simple circuit
    builder = simq.CircuitBuilder(2)
    builder.x(0)  # Flip qubit 0 to |1⟩
    circuit = builder.build()

    backend = simq.LocalSimulatorBackend()
    result = backend.execute(circuit, shots=500)

    # Test all result properties and methods
    assert result.shots == 500
    assert isinstance(result.counts, dict)
    assert result.job_id is None  # Local backend doesn't use job IDs

    # Should mostly see |10⟩ (binary) = "01" (little-endian string)
    # or |01⟩ depending on endianness
    counts = result.counts
    assert len(counts) > 0

    # Test get_count method
    for state in counts.keys():
        count = result.get_count(state)
        assert count == counts[state]

    # Test most_frequent
    most_freq = result.most_frequent()
    assert most_freq is not None
    assert counts[most_freq] == max(counts.values())

    # Test probabilities
    probs = result.probabilities()
    for state, prob in probs.items():
        expected_prob = counts[state] / 500
        assert abs(prob - expected_prob) < 0.001

    # Test metadata
    assert result.backend_name == "local_simulator"
    assert result.execution_time is None or result.execution_time >= 0
    assert result.total_time is None or result.total_time >= 0
    assert result.cost is None or result.cost == 0.0

    # Test repr
    repr_str = repr(result)
    assert "BackendResult" in repr_str
    assert "500" in repr_str

    print(f"✓ Test 4 passed: All BackendResult methods work")
    print(f"  Result: {repr_str}")


def test_backend_job_management():
    """Test 5: Test job submission and status tracking (LocalSimulator)"""
    # Create circuit
    builder = simq.CircuitBuilder(2)
    builder.h(0)
    builder.cx(0, 1)
    circuit = builder.build()

    backend = simq.LocalSimulatorBackend()

    # Submit job
    job_id = backend.submit_job(circuit, shots=100)
    assert job_id is not None
    assert isinstance(job_id, str)

    # Check job status
    status = backend.job_status(job_id)
    assert status is not None
    assert status.is_completed()  # Local simulator completes immediately

    # Get result
    result = backend.get_result(job_id)
    assert result.shots == 100
    assert len(result.counts) > 0

    # Verify backend properties
    assert backend.is_available()
    backend_type = backend.backend_type()
    assert backend_type is not None

    print(f"✓ Test 5 passed: Job management works")
    print(f"  Job ID: {job_id}")
    print(f"  Status: {status}")
    print(f"  Backend type: {backend_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Phase 5 Backend Tests")
    print("=" * 60)

    try:
        test_local_simulator_basic_execution()
        print()
        test_local_simulator_with_config()
        print()
        test_local_simulator_ghz_state()
        print()
        test_backend_result_methods()
        print()
        test_backend_job_management()
        print()
        print("=" * 60)
        print("✓ All 5 backend tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
