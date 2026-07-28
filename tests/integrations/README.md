# Tributo Integration Tests

> Run inside a Docker Ray cluster. Test scripts are executed via `docker exec` in
> the `ray-head` container, validating end-to-end workflows against real infrastructure.

---

## Test List

| Test | File | Coverage | Prerequisites |
|------|------|----------|---------------|
| MLflow E2E | `test_e2e_mlflow.py` | Synthetic data → XGBoost → MLflow tracking → ONNX export → model registry | Ray + MLflow |
| ClickHouse E2E | `test_e2e_clickhouse.py` | ClickHouse table creation / write → `load_ray_dataset_from_config` → XGBoost distributed training → MLflow → ONNX | Ray + ClickHouse + MLflow |
| Streaming | `test_e2e_streaming.py` | Streaming inference service | TBD |
| Tune | `test_e2e_tune.py` | Hyperparameter search | TBD |

---

## Environment Setup

### Start Docker Cluster

```bash
cd rayDocker && ./ray-cluster.sh up
```

Wait for all services to be healthy (~30s), then verify:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

Expected containers:

| Container | Ports | Role |
|-----------|-------|------|
| `ray-head` | 8265 / 6380 / 10001 | Ray Head (GCS + Dashboard) |
| `ray-worker-[1-3]` | — | Distributed training workers |
| `clickhouse` | 8123 (HTTP) / 9123 (Native) | OLAP database |
| `redis` | 6379 | Redis Stream message queue |
| `minio` | 9000 (S3) / 9001 (Console) | S3-compatible storage |
| `mlflow-server` | 5001 | MLflow experiment tracking |

### Test Directory Mapping

`rayDocker/docker-compose.yml` mounts the project root to `/app` inside containers.
Test files are available at `/opt/tributo/tests/integrations/` inside the container.

---

## Running Tests

All tests are executed from the host machine via `docker exec ray-head`.

### Individual Tests

```bash
# ClickHouse integration test
docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_clickhouse.py

# Redis Stream full-pipeline test
docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_redis_stream.py

# MLflow test
docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_mlflow.py
```

### Batch Run

```bash
docker exec ray-head bash -c '
  for t in /opt/tributo/tests/integrations/test_e2e_*.py; do
    echo "=== Running $t ==="
    python "$t" && echo "PASS" || echo "FAIL"
  done
'
```

### Expected Output

```
# test_e2e_clickhouse.py
ClickHouse connected ✅
MLflow connected ✅
ClickHouse table tributo_e2e_clickhouse_test: 2000 rows written
Ray cluster ready: ...
Training complete: succeeded
ONNX model: 22659 bytes
✅ ClickHouse E2E test passed

# test_e2e_redis_stream.py
Redis connected ✅
ClickHouse connected ✅
Task published: message_id=...
Ray Job submitted: ...
Event type sequence: ['PHASE', 'PHASE', 'PHASE', 'METRICS', ..., 'COMPLETED']
COMPLETED event validation passed ✅
✅ Redis Stream E2E test passed
```

---

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Docker Network                                   │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ ray-head │  │ worker-1 │  │ worker-2 │  │ worker-3 │  │  minio   │ │
│  │ 8265     │  │ 3C 4G    │  │ 3C 4G    │  │ 3C 4G    │  │ 9000 S3  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  redis   │  │clickhouse│  │ mlflow-server│  │    Host machine    │ │
│  │ 6379     │  │ 8123     │  │ 5001         │  │ docker exec trigger│ │
│  └──────────┘  └──────────┘  └──────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Redis Stream Message Format (Protocol v2.0)

**Training Task** (downstream → Tributo):

```
XADD tributo:training:tasks * \
  job_id "train-job-001" \
  payload '{"job_id":"train-job-001","algorithm":{...},...}'
```

**Training Event** (Tributo → downstream):

```
XADD tributo:aimodel:training:events:train-job-001 * \
  job_id "train-job-001" \
  payload '{"protocol_version":"2.0","event_type":"COMPLETED",...}'
```

Each stream entry contains exactly two fields: `{job_id, payload}`.
`payload` is the full JSON string of the business object — field flattening is forbidden.

---

## Troubleshooting

### Container startup failure

```bash
# View all logs
docker compose -f rayDocker/docker-compose.yml logs

# View specific service
docker compose -f rayDocker/docker-compose.yml logs clickhouse

# Rebuild cluster
cd rayDocker && ./ray-cluster.sh down && ./ray-cluster.sh up
```

### Redis connection refused

```bash
docker exec ray-head redis-cli -h redis ping
# Should return PONG
```

### ClickHouse connection refused

```bash
docker exec ray-head bash -c "
  python3 -c \"
import clickhouse_connect
c = clickhouse_connect.get_client(host='clickhouse', port=8123, username='reader', password='tributo123', database='analytics')
print(c.command('SELECT 1'))
  \"
"
```

### Ray Dashboard

Visit http://localhost:8265 to monitor training job status.
