"""Example: Basic job submission.

This example demonstrates how to submit a simple Ray job using Tributo.
"""

from tributo import JobConfig, RayJob

# Create job configuration
config = JobConfig(
    entrypoint='python -c "import ray; ray.init(); print(ray.cluster_resources())"',
    runtime_env={
        "pip": ["numpy>=1.20.0"],
    },
    num_cpus=2.0,
)

# Submit job
job = RayJob(address="http://127.0.0.1:8265", config=config)
job_id = job.submit()

print(f"Job submitted: {job_id}")

# Check status
status = job.get_status(job_id)
print(f"Job status: {status}")

# Get logs
logs = job.get_logs(job_id)
print(f"Job logs:\n{logs}")
