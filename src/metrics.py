from prometheus_client import Counter, Gauge, Histogram


BENCHMARK_PIPELINE_RUNS = Counter(
    "llm_gpu_benchmarking_pipeline_runs_total",
    "Total number of LLM GPU benchmark pipeline executions.",
    ["model_name", "target_gpu", "target_environment", "status"],
)

INFERENCE_THROUGHPUT_GAUGE = Histogram(
    "llm_gpu_benchmarking_validated_throughput_tps",
    "Tokens per second metrics captured during automated validation phases.",
    ["model_name", "target_gpu", "target_environment", "status"],
    buckets=[50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000],
)

INFERENCE_LATENCY_GAUGE = Histogram(
    "llm_gpu_benchmarking_validated_latency_ms",
    "End-to-end validation latency in milliseconds captured during automated validation phases.",
    ["model_name", "target_gpu", "target_environment", "status"],
    buckets=[25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000],
)

VALIDATION_MATRIX_BENCHMARK_THROUGHPUT_TPS = Histogram(
    "llm_gpu_benchmarking_validation_matrix_benchmark_tps",
    "Precision-mode-aware simulated benchmark throughput from the validation matrix.",
    ["model_name", "target_gpu", "target_environment", "precision_mode", "status"],
    buckets=[50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000],
)

VALIDATION_MATRIX_BENCHMARK_LATENCY_MS = Histogram(
    "llm_gpu_benchmarking_validation_matrix_benchmark_latency_ms",
    "Precision-mode-aware simulated benchmark latency from the validation matrix.",
    ["model_name", "target_gpu", "target_environment", "precision_mode", "status"],
    buckets=[25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 30000, 60000],
)

LOCAL_LLM_BENCHMARK_THROUGHPUT_TPS = Histogram(
    "llm_gpu_benchmarking_local_llm_benchmark_tps",
    "Measured local llama.cpp generation throughput from a downloaded GGUF model.",
    ["model_name", "target_gpu", "target_environment", "precision_mode", "status"],
    buckets=[1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000],
)

LOCAL_LLM_BENCHMARK_LATENCY_MS = Histogram(
    "llm_gpu_benchmarking_local_llm_benchmark_latency_ms",
    "Measured local llama.cpp end-to-end inference latency from a downloaded GGUF model.",
    ["model_name", "target_gpu", "target_environment", "precision_mode", "status"],
    buckets=[100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000, 300000],
)

PIPELINE_DURATION_SECONDS = Histogram(
    "llm_gpu_benchmarking_pipeline_duration_seconds",
    "Duration of each LLM GPU benchmarking pipeline stage.",
    ["stage", "model_name", "target_gpu", "target_environment", "precision_mode", "status"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
)

BENCHMARK_FAILURES = Counter(
    "llm_gpu_benchmarking_failures_total",
    "LLM GPU benchmarking failures by pipeline stage, reason, and model.",
    ["stage", "reason", "model"],
)

ACTIVE_WORKERS = Gauge(
    "llm_gpu_benchmarking_active_workers",
    "Current active LLM GPU benchmarking compilation or validation jobs on this worker process.",
)

VRAM_REQUIRED_GB = Gauge(
    "llm_gpu_benchmarking_vram_required_gb",
    "Estimated model VRAM requirement after applying precision mode.",
    ["model_name", "target_gpu", "precision_mode"],
)

VRAM_CAPACITY_GB = Gauge(
    "llm_gpu_benchmarking_vram_capacity_gb",
    "Available target VRAM capacity from the validation hardware profile.",
    ["model_name", "target_gpu", "precision_mode"],
)

MODEL_ACCURACY_SCORE = Gauge(
    "llm_gpu_benchmarking_validation_matrix_accuracy_score",
    "Simulated relative model quality score after applying precision mode.",
    ["model_name", "target_gpu", "precision_mode"],
)
