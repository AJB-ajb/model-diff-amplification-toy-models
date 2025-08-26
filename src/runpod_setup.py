"""
RunPod Environment Setup for Python
===================================

This module automatically configures environment variables for optimal
ML/AI workflows on RunPod, ensuring all caches use persistent storage.

Usage:
    import runpod_setup  # Auto-setup
    
    # Or explicit setup:
    import runpod_setup
    runpod_setup.setup()
"""

import os
from pathlib import Path


def setup():
    """Configure environment variables for RunPod."""
    
    # Cache directories
    cache_dirs = {
        'huggingface': '/workspace/.cache/huggingface',
        'huggingface_hub': '/workspace/.cache/huggingface/hub',
        'torch': '/workspace/.cache/torch',
        'pip': '/workspace/.cache/pip',
        'cuda': '/workspace/.cache/cuda'
    }
    
    # Create directories
    for cache_dir in cache_dirs.values():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    env_vars = {
        'HF_HOME': cache_dirs['huggingface'],
        'HF_HUB_CACHE': cache_dirs['huggingface_hub'],
        'TRANSFORMERS_CACHE': cache_dirs['huggingface_hub'],
        'TORCH_HOME': cache_dirs['torch'],
        'PIP_CACHE_DIR': cache_dirs['pip'],
        'CUDA_CACHE_PATH': cache_dirs['cuda'],
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_DOWNLOAD_TIMEOUT': '600'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    print("🚀 RunPod environment configured for Python!")
    print(f"📁 Caches redirected to /workspace/.cache/")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/workspace")
    print(f"💾 Workspace disk space: {free // (1024**3)}GB free / {total // (1024**3)}GB total")


def get_cache_dir(service='huggingface'):
    """Get the cache directory for a specific service."""
    cache_dirs = {
        'huggingface': '/workspace/.cache/huggingface',
        'torch': '/workspace/.cache/torch',
        'pip': '/workspace/.cache/pip'
    }
    return cache_dirs.get(service, '/workspace/.cache')


# Auto-setup when imported
setup()
