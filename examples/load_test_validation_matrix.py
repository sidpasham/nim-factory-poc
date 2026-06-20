import argparse
import asyncio
from itertools import cycle
from typing import Any, Dict, List

import httpx


PAYLOADS: List[Dict[str, Any]] = [
    {
        "model_name": "Llama-3-8B",
        "target_gpu": "A10G-24GB",
        "target_environment": "kubernetes",
        "precision_mode": "FP16",
    },
    {
        "model_name": "Llama-3-70B",
        "target_gpu": "A10G-24GB",
        "target_environment": "kubernetes",
        "precision_mode": "FP16",
    },
    {
        "model_name": "Llama-3-70B",
        "target_gpu": "H100-80GB",
        "target_environment": "kubernetes",
        "precision_mode": "INT4",
    },
    {
        "model_name": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "target_gpu": "NVIDIA GB200",
        "target_environment": "cloud",
        "precision_mode": "INT8",
    },
    {
        "model_name": "meta/llama-3.1-8b-instruct",
        "target_gpu": "NVIDIA T4-16GB",
        "target_environment": "on-prem",
        "precision_mode": "INT4",
    },
]


async def submit_one(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = await client.post("/factory/ingest", json=payload)
    response.raise_for_status()
    return response.json()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


async def wait_for_completion(
    client: httpx.AsyncClient,
    workflow_id: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    status_url = f"/factory/status/{workflow_id}"

    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(status_url)
        response.raise_for_status()
        status = response.json()
        if status.get("status") == "COMPLETED":
            return status
        await asyncio.sleep(1.0)

    raise TimeoutError(f"workflow {workflow_id} did not complete within {timeout_seconds}s")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Submit validation-matrix load to Model Factory.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--poll", type=parse_bool, default=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)
    payload_cycle = cycle(PAYLOADS)

    async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
        async def run_submit() -> Dict[str, Any]:
            async with semaphore:
                return await submit_one(client, next(payload_cycle))

        results = await asyncio.gather(*(run_submit() for _ in range(args.requests)))

        if args.poll:
            completed = await asyncio.gather(*(
                wait_for_completion(
                    client,
                    result["workflow_id"],
                    args.timeout_seconds,
                )
                for result in results
            ))
        else:
            completed = []

    for result in results:
        print(f"{result['workflow_id']} {result['status_url']}")

    if not args.poll:
        return

    deployable = 0
    failed = 0
    for status in completed:
        summary = status.get("pipeline_summary", {})
        final_status = summary.get("final_status", "Unknown")
        if summary.get("deployable"):
            deployable += 1
        else:
            failed += 1
        print(
            "completed "
            f"{status['workflow_id']} "
            f"final_status={final_status} "
            f"deployable={summary.get('deployable', False)}"
        )

    print(
        "summary "
        f"submitted={len(results)} completed={len(completed)} "
        f"deployable={deployable} failed={failed}"
    )


if __name__ == "__main__":
    asyncio.run(main())
