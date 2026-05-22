import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for Python/NumPy and, when available, PyTorch CPU/GPU."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception as exc:
        logger.warning("PyTorch seed setup skipped: %s", exc)
