# Ops Scripts

Shortcuts for setup, builds, and deployment.

## CUDA build

- `setup_env.sh`: sets CUDA paths and applies the 13.1 shim for `cudarc` 0.16.6.
- `check_cuda.sh`: runs `cargo check -p moshi-server --features cuda` with the shim.
- `run-moshi-server.sh`: installs and runs the server; auto-sources the shim on CUDA 13.1.

## Other utilities

- `setup-sentencepiece.sh`: builds and installs SentencePiece locally.
- `gpu-monitor.sh`: lightweight GPU usage monitoring.
- `deploy-caddy.sh`: installs/updates Caddy with the included config.
- `systemd/`: service and override files used by deployment scripts.
