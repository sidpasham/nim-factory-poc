import asyncio
import logging
import os

from prometheus_client import start_http_server
from temporalio.client import Client
from temporalio.worker import Worker

from activities import execute_compilation_and_validation
from local_llm_runtime import LocalLLMBenchmarkRunner
from workflows import LlmGpuBenchmarkingWorkflow


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)
VALIDATION_MODE_ENV = "LLM_GPU_BENCHMARKING_VALIDATION_MODE"
LOCAL_VALIDATION_MODE = "local"


async def connect_temporal() -> Client:
    temporal_address = os.getenv("TEMPORAL_ADDRESS", "temporal:7233")
    attempts = int(os.getenv("TEMPORAL_CONNECT_ATTEMPTS", "12"))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return await Client.connect(temporal_address)
        except Exception as exc:
            last_error = exc
            LOGGER.warning(
                "Temporal connection attempt %s/%s failed: %s",
                attempt,
                attempts,
                exc,
            )
            await asyncio.sleep(min(attempt, 5))

    raise RuntimeError(
        f"failed to connect to Temporal at {temporal_address} after {attempts} attempts"
    ) from last_error


async def main():
    metrics_port = int(os.getenv("METRICS_PORT", "9000"))
    start_http_server(metrics_port)

    warmup_local_llm_runtime()

    client = await connect_temporal()
    worker = Worker(
        client,
        task_queue="llm-gpu-benchmarking-task-queue",
        workflows=[LlmGpuBenchmarkingWorkflow],
        activities=[execute_compilation_and_validation],
        max_concurrent_activities=int(os.getenv("MAX_CONCURRENT_ACTIVITIES", "10")),
    )

    LOGGER.info("LLM GPU Benchmarking worker attached to Temporal; metrics_port=%s", metrics_port)
    await worker.run()


def warmup_local_llm_runtime() -> None:
    validation_mode = os.getenv(VALIDATION_MODE_ENV, LOCAL_VALIDATION_MODE).strip().lower()
    if validation_mode != LOCAL_VALIDATION_MODE:
        LOGGER.info("Skipping local LLM cache warmup for validation_mode=%s", validation_mode)
        return

    runner = LocalLLMBenchmarkRunner.from_env()
    if not runner.config.warmup_on_startup:
        LOGGER.info("Local LLM cache warmup disabled")
        return

    LOGGER.info(
        "Warming local LLM cache before accepting work; models=%s precision_modes=%s root=%s",
        ",".join(runner.config.warmup_models),
        ",".join(runner.config.warmup_precision_modes),
        runner.config.root_dir,
    )
    try:
        result = runner.warmup_cache()
        LOGGER.info(
            "Local LLM cache warmup completed in %.3fs; manifest=%s",
            result["duration_seconds"],
            result["cache_manifest_path"],
        )
    except Exception:
        if runner.config.warmup_required:
            LOGGER.exception("Local LLM cache warmup failed; failing worker startup")
            raise
        LOGGER.exception("Local LLM cache warmup failed; continuing because warmup is optional")


if __name__ == "__main__":
    asyncio.run(main())
