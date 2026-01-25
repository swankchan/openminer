# Documentation

Project documentation is centralized in `docs/` to avoid scattering notes across the repository root.

## Getting Started

- [Installation Guide](INSTALLATION.md) - Quick start, installation, configuration, and troubleshooting

## Deployment

- [Deployment Guide](DEPLOYMENT.md) - Docker deployment, reverse proxy setup, and security notes

## MinerU / vLLM

- [MinerU vLLM Setup](MINERU_VLLM_SETUP.md) - vLLM async engine configuration for improved performance

## Check Scripts (Manual Diagnostics)

Run these scripts in the `mineru2.5` conda environment:

```bash
# Check MinerU installation
python scripts/checks/check_mineru.py

# Check vLLM / mineru-vl-utils
python scripts/checks/check_vllm.py
```
