# Tributo Integration Tests

> Most end-to-end scripts run inside a Docker Ray cluster. The MLflow Hook suite
> is collected by pytest and runs against a real MLflow Tracking Server.

---

## Test List

| Test | File | Coverage | Prerequisites |
|------|------|----------|---------------|
| MLflow Hook | `test_e2e_mlflow.py` | Committed Bundle upload, replay deduplication, explicit run reuse, and failure semantics | MLflow + `registry` extra |
| ClickHouse E2E | `test_e2e_clickhouse.py` | ClickHouse table → Daft OLAP Binding → explicit Daft-to-Ray adapter → XGBoost distributed training → MLflow → ONNX | Ray + Daft + `daft-olap-connectors` + ClickHouse + MLflow |
| Dual-engine Docker | `test_data_ingestion_dual_engine.py` | Local Parquet, full ETL chain, typed handles, worker-version evidence | Docker Ray cluster + Daft |
| File conformance | `../integration/test_data_ingestion_conformance.py` | Local/MinIO Parquet and CSV through Ray Data and Daft | Local Ray runtime + MinIO |
| Table conformance | `../integration/test_table_format_ingestion.py` | Local/MinIO Iceberg and Lance through Ray Data and Daft | Local Ray runtime + MinIO |
| PostgreSQL conformance | `../integration/test_postgresql_ingestion.py` | Structured table read through Ray Data and Daft | Local Ray runtime + PostgreSQL |
| Streaming | `test_e2e_streaming.py` | Streaming inference service | TBD |
| Tune | `test_e2e_tune.py` | Hyperparameter search | TBD |

---

## MLflow Hook Suite

The Hook suite validates the committed-bundle integration rather than automatic
Model Registry or Stage behavior. Missing infrastructure is a test failure, not
a reason to skip the suite.

Start and verify the existing local service:

```bash
docker start pista-mlflow-server
curl --fail 'http://127.0.0.1:8050/api/2.0/mlflow/experiments/search?max_results=1'
```

Run the dedicated suites:

```bash
uv run --extra registry pytest tests/integrations/test_e2e_mlflow.py \
  -m integration -vv
uv run --extra registry pytest tests/registry/test_integration.py \
  -m integration -vv
```

Set `MLFLOW_TRACKING_URI` to use another real server. The default is
`http://127.0.0.1:8050`.

---

## Distributed Test Environment

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

## Running Distributed Tests

The remaining end-to-end scripts are executed from the host machine via
`docker exec ray-head`.

### Individual Tests

```bash
# ClickHouse integration test
docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_clickhouse.py

# Redis Stream full-pipeline test
docker exec ray-head python /opt/tributo/tests/integrations/test_e2e_redis_stream.py

# Required Docker-cluster ingestion slice
docker exec ray-head env TRIBUTO_DOCKER_RAY_TEST=1 \
  python -m tests.integrations.test_data_ingestion_dual_engine
```

The ClickHouse and multi-class scripts require the independently installed
`daft-olap-connectors` distribution. They use an explicit conversion adapter;
the Gateway itself never changes the selected engine or disguises a Daft
DataFrame as a Ray Dataset.

The PR workflow runs two required ingestion gates. The semantic gate executes
the file, table-format, and PostgreSQL Conformance files with locked `data`,
`data-daft`, and `postgresql` extras. The distributed gate builds the isolated
`Dockerfile.data-ingestion` image, starts one Ray head, one Ray worker, and
MinIO with `docker-compose.data-ingestion.yml`, initializes the shared volume
for the unprivileged `ray` user, then runs this module with the Daft Ray runner
and mandatory Driver/Worker version evidence. Both jobs feed `core-gate`;
infrastructure absence is a failure rather than a passing skip.

### Batch Run

```bash
docker exec ray-head bash -c '
  for t in /opt/tributo/tests/integrations/test_e2e_*.py; do
    if [ "$(basename "$t")" = "test_e2e_mlflow.py" ]; then
      continue
    fi
    echo "=== Running $t ==="
    python "$t" && echo "PASS" || echo "FAIL"
  done
'
```

The batch loop excludes `test_e2e_mlflow.py` because that module is a pytest
suite and must be run with the fail-fast command above.

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
│                         Docker Network                                  │
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
