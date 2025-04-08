import sys
import torch
import torch_geometric
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_setup():
    """Check if the ML environment is properly set up."""
    # Check Python version
    python_version = sys.version.split()[0]
    logger.info(f"Python version: {python_version}")
    
    # Check PyTorch
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {cuda_available}")
    logger.info(f"Device: {device}")
    
    # Check PyTorch Geometric
    logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
    
    # Check NumPy
    logger.info(f"NumPy version: {np.__version__}")
    
    # Try creating a small tensor
    try:
        x = torch.randn(3, 3).to(device)
        logger.info("✓ Successfully created and moved tensor to device")
    except Exception as e:
        logger.error(f"Failed to create tensor: {e}")
        return False
    
    logger.info("✓ Environment setup looks good!")
    return True

if __name__ == "__main__":
    check_setup() 