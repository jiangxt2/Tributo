"""Create a small multi-file Parquet fixture in the test HDFS cluster."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _run(*args: str) -> None:
    subprocess.run(args, check=True, text=True)


def main() -> None:
    local_root = Path("/workspace/tributo-work/hdfs-fixture")
    local_root.mkdir(parents=True, exist_ok=True)
    hdfs_root = "/tributo-hdfs/parquet"
    _run("hdfs", "dfs", "-mkdir", "-p", hdfs_root)
    _run("hdfs", "dfs", "-chmod", "-R", "777", "/tributo-hdfs")

    for index in range(8):
        path = local_root / f"part-{index}.parquet"
        table = pa.table(
            {
                "id": list(range(index * 200, (index + 1) * 200)),
                "part": [index] * 200,
            }
        )
        pq.write_table(table, path)
        _run("hdfs", "dfs", "-put", "-f", str(path), f"{hdfs_root}/{path.name}")


if __name__ == "__main__":
    main()
