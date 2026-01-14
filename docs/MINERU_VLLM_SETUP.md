# MinerU vLLM-Async-Engine Setup Guide

## Overview

The application supports running MinerU 2.5 via `vllm-async-engine`, which can provide better throughput (reported up to ~2.12 fps concurrent inference).

## Installation

### 1. Install mineru-vl-utils and vllm

```bash
# In the mineru2.5 environment
conda activate mineru2.5

# Install the vllm extra (recommended)
pip install "mineru-vl-utils[vllm]"

# Or install separately
pip install mineru-vl-utils vllm
```

### 2. Configure settings

Set values in `config.py` or `.env`:

```python
# config.py
USE_VLLM_ASYNC = True  # Enable vllm-async-engine
MINERU_MODEL_NAME = "opendatalab/MinerU2.5-2509-1.2B"  # Model name
MINERU_DEVICE = "cuda"  # Or "cpu", "cuda:0", etc.
```

Or in `.env`:
```env
USE_VLLM_ASYNC=True
MINERU_MODEL_NAME=opendatalab/MinerU2.5-2509-1.2B
MINERU_DEVICE=cuda
```

## Usage

### Enable vllm-async-engine (recommended)

With `USE_VLLM_ASYNC=True`, the app will:
1. Load the model at startup
2. Process multiple PDFs concurrently using async execution
3. Provide improved throughput

### CLI mode (fallback)

If `USE_VLLM_ASYNC=False` or initialization fails, the app automatically falls back to CLI mode.

## Notes

### Automatic fallback

- If `vllm-async-engine` is unavailable or fails to initialize, the app uses CLI mode
- This keeps the app working in more environments

### Performance

- **vllm-async-engine**: can reach ~2.12 fps concurrent inference (on A100)
- **CLI mode**: slower but typically more stable

### Debug output

When `DEBUG=True`, logs include:
- MinerU initialization status
- Processing progress
- Error messages

## Requirements

### vllm-async-engine requirements

- GPU: CUDA-capable GPU is recommended
- VRAM: at least 8GB (16GB+ recommended)
- Python: 3.8+
- CUDA: 11.8+ (if using GPU)

### CPU mode

If you do not have a GPU:
```env
MINERU_DEVICE=cpu
```

Performance will be significantly slower.

## Troubleshooting

### 1. Model download

On first use, the model is downloaded from Hugging Face. Ensure:
- Working internet connection
- Enough disk space (about 2-3GB for the model)

### 2. CUDA errors

If you see CUDA errors:
- Check CUDA version compatibility
- Try `MINERU_DEVICE=cpu` (slower but stable)

### 3. Out of memory

If you run out of memory:
- Reduce concurrent requests
- Use a smaller batch size
- Consider using CLI mode

### 4. Initialization failure

If initialization fails:
- Set `DEBUG=True` to see detailed errors
- The app will fall back to CLI mode
- Ensure all dependencies are installed

## Performance Tips

1. Use GPU: set `MINERU_DEVICE=cuda`
2. Batch processing: processing multiple PDFs at once can better utilize concurrency
3. Model cache: the model is cached and subsequent runs are faster

## References

- [MinerU2.5 on Hugging Face](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B)
- [mineru-vl-utils on GitHub](https://github.com/opendatalab/mineru-vl-utils)

