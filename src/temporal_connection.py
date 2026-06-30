import asyncio
import logging
import os
from typing import Iterable

from temporalio.client import Client


TEMPORAL_ADDRESS_ENV = "TEMPORAL_ADDRESS"
TEMPORAL_SERVICE_HOST_ENV = "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_HOST"
TEMPORAL_SERVICE_PORT_ENV = "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_PORT_GRPC"
TEMPORAL_SERVICE_PORT_FALLBACK_ENV = "LLM_GPU_BENCHMARKING_TEMPORAL_SERVICE_PORT"


def temporal_address_candidates(default_address: str) -> list[str]:
    candidates = [
        os.getenv(TEMPORAL_ADDRESS_ENV, default_address).strip() or default_address,
    ]

    service_host = os.getenv(TEMPORAL_SERVICE_HOST_ENV, "").strip()
    service_port = (
        os.getenv(TEMPORAL_SERVICE_PORT_ENV, "").strip()
        or os.getenv(TEMPORAL_SERVICE_PORT_FALLBACK_ENV, "").strip()
    )
    if service_host:
        candidates.append(f"{service_host}:{service_port or '7233'}")

    candidates.append(default_address)
    return _unique(candidates)


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


async def connect_temporal_client(
    *,
    default_address: str,
    attempts: int,
    logger: logging.Logger,
) -> Client:
    last_error: Exception | None = None
    addresses = temporal_address_candidates(default_address)

    for address in addresses:
        for attempt in range(1, attempts + 1):
            try:
                return await Client.connect(address)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Temporal connection attempt %s/%s failed for %s: %s",
                    attempt,
                    attempts,
                    address,
                    exc,
                )
                await asyncio.sleep(min(attempt, 5))

    raise RuntimeError(
        "failed to connect to Temporal after trying "
        f"{', '.join(addresses)}"
    ) from last_error
