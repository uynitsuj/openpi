# SkyPilot Instructions

## Setup
Install skypilot
```
pip install awscli
pip install "skypilot-nightly[kubernetes,aws,gcp,azure,oci,lambda,runpod,fluidstack,paperspace,cudo,ibm,scp,nebius]"
```
Configure aws access keys
```
aws configure
```

Check sky enabled infra
```
sky check
```
should look like:
```
🎉 Enabled infra 🎉
  AWS [compute, storage]

To enable a cloud, follow the hints above and rerun: sky check
If any problems remain, refer to detailed docs at: https://docs.skypilot.co/en/latest/getting-started/installation.html

Using SkyPilot API server: http://127.0.0.1:46580
```

## PI0 Train Instructions

Launch training with appropriate config name (should reflect local openpi workdir)
```
uv run sky/launch_training.py --config-name pi0_xmi_rby_low_mem_finetune
```