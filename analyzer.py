import time
from functools import wraps
import numpy as np
from numba import njit

# 1. Bring back our custom timing decorator from eariler!
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000 # Convert to milliseconds
        print(f"⏱️ [{func.__name__}] took {duration:.3f} ms")
        return result
    return wrapper


# --- STAGE 1: File Ingestion & Parsing (The Real-World Bottleneck) ---
@measure_time
def parse_log_file(file_path):
    """Simulate reading raw text logs and extracting numeric latency values."""
    latencies = []
    # Simulating reading lines from a file
    with open(file_path, "r") as f:
        for line in f:
            # Parse CSV-style log: "timestamp, latency_ms\n"
            parts = line.strip().split(",")
            if len(parts) == 2:
                latencies.append(float(parts[1]))
    return latencies


# --- STAGE 2: Optimized Numerical Processing (Numba Kernel) ---
@njit(fastmath=True)
def _compute_metrics_kernel(values):
    """Warned Numba JIT kernel for high-performance anomally scoring."""
    result = np.empty_like(values)
    for i in range(len(values)):
        v = values[i]
        # Example metric formula: penalty score based on response time
        result[i] = (v * v) / 1000.0 + v
    return result


@measure_time
def process_metrics_batch(raw_list):
    """Respects the execution boundary: converts list to array once, computes, returns array."""
    # Convert list to typed numpy array (The contract boundary)
    array_data = np.asarray(raw_list, dtype=np.float64)
    return _compute_metrics_kernel(array_data)

# --- APPROACH A: Standard Python List Comprehension ---
@measure_time
def process_latencies_list(values):
    # Simulating a heavy per-element telemetry transformation
    return [v * v + v for v in values]

# --- APPROACH B: NumPy Vectorized Array ---
@measure_time
def process_latencies_numpy(array_values):
    # Batch processed natively via C-implemented ufuncs
    return array_values * array_values + array_values


# --- APPROACH C: Numba JIT-Compiled Loop ---
@njit(fastmath=True)
def _numba_kernel(values):
    result = np.empty_like(values)
    for i in range(len(values)): 
        v = values[i]
        result[i] = v * v + v
    return result

@measure_time
def process_latencies_numba(array_values):
    return _numba_kernel(array_values)

if __name__ == "__main__":
    mock_file = "server_latencies.log"

    # 1. Generate a mock log file with 500,000 records
    print("Generating mock log file on disk...")
    with open(mock_file, "w") as f:
        for i in range(500_000):
            f.write(f"2026-10-26T12:00:00Z, {i % 400 + 15.5}\n")

    print("\n--- Running End-to-End LogPulse Pipeline ---")


    # Generate 1 million server response latency values
    print("Generating 1,000,000 telemetry records...")
    raw_list = [float(i % 500) for i in range(1_000_000)]
    raw_array = np.asarray(raw_list, dtype=np.float64)

    # Warm up Numba JIT complier (so compilation time isn't counted in execution)
    _numba_kernel(raw_array[:10])

    print("\n--- Running Performance Benchmarks ---")
    res_list = process_latencies_list(raw_list)
    res_numpy = process_latencies_numpy(raw_array)
    res_numba = process_latencies_numba(raw_array)

    # Verify accuracy
    np.testing.assert_array_equal(res_numpy, res_numba)
    print("\n✅ All performance outputs match successfully!")

    # Measure string parsing and I/O stage
    parsed_list = parse_log_file(mock_file)

    # Warm up Numba complier
    _compute_metrics_kernel(np.array([10.0, 20.0], dtype=np.float64))