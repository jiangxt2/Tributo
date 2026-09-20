"""Create deterministic local Parquet input for the Core quickstart."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

output_dir = Path("tributo-quickstart").resolve()
output_dir.mkdir(parents=True, exist_ok=True)
data_path = output_dir / "input.parquet"

table = pa.table(
    {
        "message_count": [1, 2, 1, 8, 9, 7, 2, 10],
        "call_duration": [2, 1, 3, 9, 8, 10, 2, 9],
        "label": [0, 0, 0, 1, 1, 1, 0, 1],
    }
)
pq.write_table(table, data_path)

print(f"Wrote {data_path}")
