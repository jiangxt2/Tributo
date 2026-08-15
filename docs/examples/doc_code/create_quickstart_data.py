"""Create deterministic local input and a formal Tributo execution request."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

output_dir = Path("tributo-quickstart").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
data_path = output_dir / "training.parquet"
bundle_path = output_dir / "bundle"
config_path = output_dir / "execution.json"

table = pa.table(
    {
        "message_count": [1, 2, 1, 8, 9, 7, 2, 10],
        "call_duration": [2, 1, 3, 9, 8, 10, 2, 9],
        "label": [0, 0, 0, 1, 1, 1, 0, 1],
    }
)
pq.write_table(table, data_path)

request = {
    "algorithm": "multinomial_nb",
    "profile": "local",
    "worker_count": 2,
    "input": {
        "ingestion": {
            "source": {"type": "parquet", "path": str(data_path)},
            "engine": "ray",
        },
        "features": ["message_count", "call_duration"],
        "label": "label",
    },
    "algorithm_config": {
        "alpha": 1.0,
        "output": {"bundle_uri": str(bundle_path)},
    },
    "local_runtime": {"num_cpus": 2, "num_gpus": 0},
}
config_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

print(f"Wrote {data_path}")
print(f"Wrote {config_path}")
