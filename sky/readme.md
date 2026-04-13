# SkyPilot Instructions

## Setup
Install skypilot and awscli via the `sky` dependency group:
```
uv sync --group sky
```
### To configure AWS access keys
```
uv run aws configure
```
### To configure Lambda access keys
To configure credentials, go to: https://cloud.lambdalabs.com/api-keys to generate API key and add the line
```
api_key = [YOUR API KEY]
```
to ~/.lambda_cloud/lambda_keys

Check sky enabled infra
```
uv run sky check
```
should look something like:
```
🎉 Enabled infra 🎉
  AWS [compute, storage]
  Lambda [compute]

To enable a cloud, follow the hints above and rerun: `uv sky check`
If any problems remain, refer to detailed docs at: https://docs.skypilot.co/en/latest/getting-started/installation.html

Using SkyPilot API server: http://127.0.0.1:46580
```

## PI0 Train Instructions

Launch training with appropriate config name (should reflect local openpi workdir)
```
uv run sky/launch_training.py --config-name pi0_xmi_rby_low_mem_finetune
```

By default, `--managed` is enabled which uses `sky jobs launch` -- this guarantees
the cluster is torn down when the job finishes (or fails), preventing leaked GPU
instances. To use a persistent cluster instead:
```
uv run sky/launch_training.py --config-name pi0_xmi_rby_low_mem_finetune --no-managed
```

### Dry run

Preview the generated SkyPilot YAML without launching:
```
uv run sky/launch_training.py --config-name pi0_xmi_rby_low_mem_finetune --dry-run
```

## Managing SkyPilot Instances

### Cluster lifecycle

```bash
# List all clusters and their status
sky status

# Stop a cluster (pauses billing, keeps disk -- can restart later)
sky stop <cluster-name>

# Restart a stopped cluster
sky start <cluster-name>

# Terminate a cluster permanently (frees all resources)
sky down <cluster-name>

# Terminate ALL clusters
sky down -a

# Set autostop (auto-terminate after N minutes idle)
sky autostop <cluster-name> -i 10 --down

# Cancel a pending autostop
sky autostop <cluster-name> --cancel
```

### Job operations on clusters

```bash
# List jobs running on a cluster
sky queue <cluster-name>

# Stream logs for a specific job
sky logs <cluster-name> <job-id>

# Cancel a specific job on a cluster
sky cancel <cluster-name> <job-id>

# Cancel all jobs on a cluster
sky cancel <cluster-name> --all
```

### Managed jobs (recommended -- auto-teardown on completion)

```bash
# List all managed jobs
sky jobs queue

# Stream logs for a managed job
sky jobs logs <job-id>

# Cancel a managed job
sky jobs cancel <job-id>

# Cancel by name
sky jobs cancel -n <job-name>
```

### Debugging and utility

```bash
# SSH into a cluster's head node
ssh <cluster-name>

# Check available GPU types and pricing
sky show-gpus A100-80GB:8 --infra lambda/us-west-2

# Verify cloud credentials
sky check

# Web dashboard (if API server is running)
sky dashboard
```

### Common cleanup workflow

If a training run finishes but the cluster is still alive:
```bash
# Check what's running
sky status
sky queue <cluster-name>

# If no active jobs, tear it down
sky down <cluster-name>
```

To prevent this from happening, use `--managed` (default) which uses `sky jobs launch`
for guaranteed auto-teardown, or set autostop:
```bash
sky autostop <cluster-name> -i 10 --down
```
