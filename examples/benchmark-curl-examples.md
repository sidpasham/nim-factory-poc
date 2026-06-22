# Benchmark Curl Examples

Use these one at a time after the local Kubernetes Helm runtime path is running.

The local run scripts write `.runtime.env` with the correct `BASE_URL`.
Load it before running examples:

```bash
source .runtime.env
echo "${BASE_URL}"
```

`POST /benchmarks` starts a Temporal workflow and returns a `workflow_id`.
Poll the returned `status_url` to fetch the completed `pipeline_summary`.

## Quick Checks

Use these before running model validation calls.

```bash
curl -sS "${BASE_URL}/health" | jq .

curl -sS "${BASE_URL}/metrics" | head -40
```

## Recommended Demo Sweep

These calls exercise different model, hardware, and environment labels while
keeping the run count small enough for a live demo.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes","precision_mode":"INT8"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Llama-3-70B","target_gpu":"A10G-24GB","target_environment":"kubernetes","precision_mode":"FP16"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Llama-3-70B","target_gpu":"H100-80GB","target_environment":"kubernetes","precision_mode":"INT4"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI355X","target_environment":"on-prem"}' \
  | jq .
```

## One Run Per Model

Use these when you want Grafana to show all configured LLM models.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"z-ai/glm-5.1","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"minimaxai/minimax-m2.7","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"minimaxai/minimax-m3","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"deepseek-ai/deepseek-v4-flash","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .
```

## Hardware Sweep

Use one stable model across every configured hardware target so Grafana can
compare TPS by hardware label.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB300","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA B200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI355X","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI455X","target_environment":"kubernetes"}' \
  | jq .
```

## Model and Hardware Comparison

Use these to populate Grafana with model-to-hardware comparison data without
running every possible combination.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB300","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA B200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI355X","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"z-ai/glm-5.1","target_gpu":"AMD MI455X","target_environment":"kubernetes"}' \
  | jq .
```

## Environment Sweep

Use one model and hardware target across each benchmark environment label.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"cloud"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"on-prem"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .
```

## Compact Loop Sweeps

These generate multiple runs quickly. They are useful for filling Grafana in
the default validation-matrix mode.

```bash
for hardware in "NVIDIA GB200" "NVIDIA GB300" "NVIDIA B200" "AMD MI355X" "AMD MI455X"; do
  curl -sS -X POST "${BASE_URL}/benchmarks" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\":\"meta/llama-3.1-8b-instruct\",\"target_gpu\":\"${hardware}\",\"target_environment\":\"kubernetes\",\"precision_mode\":\"INT8\"}" \
    | jq '{workflow_id, status_url}'
done

for environment in "cloud" "on-prem" "kubernetes"; do
  curl -sS -X POST "${BASE_URL}/benchmarks" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\":\"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning\",\"target_gpu\":\"NVIDIA GB200\",\"target_environment\":\"${environment}\",\"precision_mode\":\"INT4\"}" \
    | jq '{workflow_id, status_url}'
done

for model in \
  "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" \
  "meta/llama-3.1-8b-instruct" \
  "z-ai/glm-5.1"; do
  curl -sS -X POST "${BASE_URL}/benchmarks" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\":\"${model}\",\"target_gpu\":\"NVIDIA GB200\",\"target_environment\":\"kubernetes\",\"precision_mode\":\"INT8\"}" \
    | jq '{workflow_id, status_url}'
done
```

## Failure-Path Examples

These intentionally exercise validation-matrix success and failure routing.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Llama-3-70B","target_gpu":"A10G-24GB","target_environment":"kubernetes","precision_mode":"FP16"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Llama-3-70B","target_gpu":"H100-80GB","target_environment":"kubernetes","precision_mode":"INT4"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"Llama-3-30B","target_gpu":"T4-16GB","target_environment":"on-prem","precision_mode":"FP16"}' \
  | jq .
```

## Validation Error Examples

These do not call NVIDIA because the benchmark rejects unsupported input before
model validation.

```bash
curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA-HOSTED","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/benchmarks" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB200","target_environment":"bare-metal"}' \
  | jq .
```
