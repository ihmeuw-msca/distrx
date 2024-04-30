from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt

def log(mu: npt.ArrayLike, sigma: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    mu_trans = np.log(mu)
    sigma_trans = sigma * (1.0 / sigma)