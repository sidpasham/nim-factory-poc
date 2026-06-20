from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import execute_compilation_and_validation


@workflow.defn
class ModelFactoryWorkflow:
    @workflow.run
    async def run(self, state: dict) -> dict:
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=3,
            non_retryable_error_types=[
                "UnsupportedProvisioningTargetError",
                "ValueError",
            ],
        )

        return await workflow.execute_activity(
            execute_compilation_and_validation,
            state,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry_policy,
        )
