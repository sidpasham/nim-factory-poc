# Benchmark Curl Examples

Use these after the local Kubernetes Helm runtime path is running. The default
validation mode is `local`, so requests must use a model configured in
`src/config/local_llm_profiles.json`.

The local run scripts write `.runtime.env` with the correct `BASE_URL`.

```bash
source .runtime.env
echo "${BASE_URL}"
```

`POST /benchmarks` starts a Temporal workflow and returns a `workflow_id`.
Poll the returned `status_url` to fetch the completed `pipeline_summary`.

## Quick Checks

```bash
curl -sS "${BASE_URL}/health" | jq .

curl -sS "${BASE_URL}/metrics" | head -40
```

## Local LLM Runs

First local run may build llama.cpp and download the GGUF artifact, so it can
take several minutes. Later runs reuse `.runtime/local-llm`.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Qwen/Qwen2.5-0.5B-Instruct-GGUF","target_gpu":"A10G-24GB","target_environment":"kubernetes","precision_mode":"INT4"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Qwen/Qwen2.5-0.5B-Instruct-GGUF","target_gpu":"H100-80GB","target_environment":"kubernetes","precision_mode":"INT8"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Qwen/Qwen2.5-0.5B-Instruct-GGUF","target_gpu":"NVIDIA GB200","target_environment":"cloud","precision_mode":"FP16"}' \
  | jq .
```

## Alias Examples

These names map to the same configured Qwen profile.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen2.5-0.5b-instruct","target_gpu":"AMD MI355X","target_environment":"on-prem","precision_mode":"INT8"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"local-mac-m3-qwen2.5-0.5b","target_gpu":"NVIDIA T4-16GB","target_environment":"kubernetes","precision_mode":"INT4"}' \
  | jq .
```

## Poll A Workflow

```bash
workflow_id="<workflow_id_from_post_response>"
curl -sS "${BASE_URL}/benchmarks/${workflow_id}" | jq .
```

## Local Failure Examples

These fail without falling back to a smaller model or the deterministic matrix.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Llama-3-70B","target_gpu":"A10G-24GB","target_environment":"kubernetes","precision_mode":"INT4"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Qwen/Qwen2.5-0.5B-Instruct-GGUF","target_gpu":"NVIDIA GB200","target_environment":"bare-metal","precision_mode":"INT4"}' \
  | jq .
```

## Explicit Matrix Mode

The old deterministic matrix backend is still available only by explicit
configuration. Use it for fast routing demos, not for production-like local
benchmarking.

```bash
LLM_GPU_BENCHMARKING_VALIDATION_MODE=matrix ./scripts/local-helm-deployment.sh up
```
