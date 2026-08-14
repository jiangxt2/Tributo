# Vector indexing

Build, search, optimize, and compact Lance vector indexes through Ray Jobs.
This Alpha component starts from an existing Lance dataset with a fixed-size
vector column.

```{toctree}
:maxdepth: 1

getting-started
key-concepts
user-guides/build-search
```

## Supported operations

| Operation | Contract |
| --- | --- |
| Build | IVF_FLAT or IVF_PQ with l2, cosine, or dot distance |
| Search | Fixed-version global Top-K with inline or Parquet delivery |
| Optimize | Incrementally index appended fragments on the active dataset version |
| Compact | Delegate distributed Lance file compaction and record coverage evidence |

Tributo validates the control plane and records evidence. Lance owns dataset
metadata and transactions. Lance-Ray owns distributed index construction,
search, and maintenance tasks. Ray owns scheduling and worker resources.

Install the fixed compatibility profile with
`python -m pip install "tributo[vector-index]"`.

See the generated
[Vector-index API](../reference/api/vector-index.md) for
public signatures and stability labels.
