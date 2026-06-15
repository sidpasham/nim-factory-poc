# Factory Ingest Curl Examples

Use these one at a time after the local Docker Compose stack is running.

These examples call:

```text
http://localhost:8000
```

If you are testing the Kubernetes Helm demo through port-forwarding, replace
`localhost:8000` with `localhost:8001`.


## Recommended Demo Sweep

These calls exercise different model, hardware, and environment labels while
keeping the run count small enough for a live demo.

```bash
curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB300","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"z-ai/glm-5.1","target_gpu":"NVIDIA B200","target_environment":"cloud"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI355X","target_environment":"on-prem"}' \
  | jq .
```

## Hardware Sweep

Use one stable model across every configured hardware target so Grafana can
compare TPS by hardware label.

```bash
curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA GB300","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"NVIDIA B200","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI355X","target_environment":"kubernetes"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"meta/llama-3.1-8b-instruct","target_gpu":"AMD MI455X","target_environment":"kubernetes"}' \
  | jq .
```

## Environment Sweep

Use one model and hardware target across each factory environment label.

```bash
curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"cloud"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"on-prem"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","target_gpu":"NVIDIA GB200","target_environment":"kubernetes"}' \
  | jq .
```

## Failure-Path Examples

These larger model profiles may be slower or time out depending on the NVIDIA
hosted endpoint response time. They are useful when you want Grafana to show
both success and failure routing.

```bash
curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"minimaxai/minimax-m2.7","target_gpu":"AMD MI355X","target_environment":"on-prem"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"minimaxai/minimax-m3","target_gpu":"AMD MI455X","target_environment":"cloud"}' \
  | jq .

curl -sS -X POST http://localhost:8000/factory/ingest \
  -H "Content-Type: application/json" \
  -d '{"model_name":"deepseek-ai/deepseek-v4-flash","target_gpu":"NVIDIA GB200","target_environment":"on-prem"}' \
  | jq .
```
