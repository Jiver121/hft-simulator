"""
Advanced AI/ML Trading Intelligence Layer

This module provides state-of-the-art machine learning capabilities for high-frequency trading,
including deep learning models, reinforcement learning agents, ensemble methods, and more.

Components:
- Deep Learning: LSTM, GRU, Transformer models for price prediction
- Reinforcement Learning: PPO, A3C, SAC agents for adaptive trading
- Ensemble Methods: Model combination and voting systems
- Feature Store: Real-time feature engineering with 500+ features
- GPU Acceleration: CUDA-optimized training and inference
- Hyperparameter Tuning: Automated optimization with Optuna/Ray Tune
- Model Versioning: MLFlow-based model lifecycle management
- Explainable AI: SHAP, LIME for model interpretability
- Anomaly Detection: Market regime change detection
- Sentiment Analysis: News and social media sentiment integration
"""

from typing import Dict, Any, Optional
import warnings
import os
import logging

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU availability check
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        logger.info(f"GPU acceleration available: {GPU_NAME}")
    else:
        logger.info("No GPU detected, falling back to CPU")
except ImportError:
    GPU_AVAILABLE = False
    GPU_COUNT = 0
    GPU_NAME = None
    logger.warning("PyTorch not available, GPU acceleration disabled")

# TensorFlow GPU check
try:
    import tensorflow as tf
    TF_GPU_AVAILABLE = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    if TF_GPU_AVAILABLE:
        logger.info("TensorFlow GPU support detected")
except ImportError:
    TF_GPU_AVAILABLE = False
    logger.warning("TensorFlow not available")

# Component availability registry
COMPONENTS = {
    'deep_learning': True,
    'reinforcement': True,
    'ensemble': True,
    'feature_store': True,
    'gpu': GPU_AVAILABLE,
    'hyperparameter_tuning': True,
    'model_versioning': True,
    'explainable_ai': True,
    'anomaly_detection': True,
    'sentiment_analysis': True,
}

# Version info
__version__ = "1.0.0"
__author__ = "HFT ML Team"

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information for ML components."""
    return {
        'version': __version__,
        'gpu_available': GPU_AVAILABLE,
        'gpu_count': GPU_COUNT,
        'gpu_name': GPU_NAME,
        'tf_gpu_available': TF_GPU_AVAILABLE,
        'components': COMPONENTS,
        'cuda_version': torch.version.cuda if GPU_AVAILABLE else None,
        'torch_version': torch.__version__ if 'torch' in locals() else None,
        'tf_version': tf.__version__ if 'tf' in locals() else None,
    }

def initialize_ml_environment(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize the ML environment with optimal settings."""
    config = config or {}
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    seed = config.get('random_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    if GPU_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set optimal GPU settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if TF_GPU_AVAILABLE:
        tf.random.set_seed(seed)
        # Configure TensorFlow GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
    
    # Set threading for optimal performance
    os.environ['OMP_NUM_THREADS'] = str(config.get('num_threads', 4))
    os.environ['MKL_NUM_THREADS'] = str(config.get('num_threads', 4))
    
    logger.info("ML environment initialized successfully")
    return get_system_info()

# Export commonly used imports
__all__ = [
    'get_system_info',
    'initialize_ml_environment',
    'GPU_AVAILABLE',
    'GPU_COUNT',
    'COMPONENTS',
    '__version__',
]
