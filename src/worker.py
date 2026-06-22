import asyncio
import logging
import os

from prometheus_client import start_http_server
from temporalio.client import Client
from temporalio.worker import Worker

from activities import execute_compilation_and_validation
from workflows import LlmGpuBenchmarkingWorkflow


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger(__name__)


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


if __name__ == "__main__":
    asyncio.run(main())
