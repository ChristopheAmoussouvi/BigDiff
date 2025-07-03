"""
Differential Privacy implementation for numerical data anonymization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    Differential Privacy implementation with multiple noise mechanisms.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize DifferentialPrivacy.
        
        Args:
            random_seed: Optional seed for reproducible results
        """
        if random_seed:
            np.random.seed(random_seed)
        
        self.epsilon_used = 0.0  # Privacy budget tracking
        
    def apply_laplace_noise(self, 
                           data: pd.Series, 
                           epsilon: float,
                           sensitivity: Optional[float] = None,
                           clipping_bounds: Optional[Tuple[float, float]] = None) -> pd.Series:
        """
        Apply Laplace noise for pure differential privacy.
        
        Args:
            data: Input data series
            epsilon: Privacy budget
            sensitivity: Global sensitivity (calculated if not provided)
            clipping_bounds: Optional bounds for clipping (min, max)
            
        Returns:
            Noisy data series
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Handle missing values
        if data.isnull().any():
            logger.warning(f"Data has missing values, filling with median")
            data = data.fillna(data.median())
        
        # Apply clipping if bounds specified
        if clipping_bounds:
            min_val, max_val = clipping_bounds
            clipped_values = np.clip(np.array(data.values), min_val, max_val)
            data = pd.Series(clipped_values, index=data.index)
            sensitivity = max_val - min_val
        elif sensitivity is None:
            # Use robust sensitivity calculation
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            sensitivity = 1.5 * iqr if iqr > 0 else data.std()
        
        # Apply Laplace noise
        scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=len(data))
        noisy_values = np.array(data.values) + noise
        noisy_data = pd.Series(noisy_values, index=data.index)
        
        # Track privacy budget
        self.epsilon_used += epsilon
        
        logger.info(f"Applied Laplace noise with ε={epsilon}, sensitivity={sensitivity:.3f}")
        
        return noisy_data
    
    def apply_gaussian_noise(self, 
                            data: pd.Series, 
                            epsilon: float,
                            delta: float = 1e-5,
                            sensitivity: Optional[float] = None,
                            clipping_bounds: Optional[Tuple[float, float]] = None) -> pd.Series:
        """
        Apply Gaussian noise for (ε,δ)-differential privacy.
        
        Args:
            data: Input data series
            epsilon: Privacy budget
            delta: Failure probability
            sensitivity: Global sensitivity
            clipping_bounds: Optional bounds for clipping
            
        Returns:
            Noisy data series
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("Delta must be between 0 and 1")
        
        # Handle missing values
        if data.isnull().any():
            logger.warning(f"Data has missing values, filling with median")
            data = data.fillna(data.median())
        
        # Apply clipping if bounds specified
        if clipping_bounds:
            min_val, max_val = clipping_bounds
            clipped_values = np.clip(np.array(data.values), min_val, max_val)
            data = pd.Series(clipped_values, index=data.index)
            sensitivity = max_val - min_val
        elif sensitivity is None:
            # Use robust sensitivity calculation
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            sensitivity = 1.5 * iqr if iqr > 0 else data.std()
        
        # Calculate noise scale for Gaussian mechanism
        c = np.sqrt(2 * np.log(1.25 / delta))
        sigma = c * sensitivity / epsilon
        
        # Apply Gaussian noise
        noise = np.random.normal(loc=0, scale=sigma, size=len(data))
        noisy_values = np.array(data.values) + noise
        noisy_data = pd.Series(noisy_values, index=data.index)
        
        # Track privacy budget
        self.epsilon_used += epsilon
        
        logger.info(f"Applied Gaussian noise with ε={epsilon}, δ={delta}, sensitivity={sensitivity:.3f}")
        
        return noisy_data
    
    def calculate_sensitivity(self, data: pd.Series, method: str = "robust") -> float:
        """
        Calculate sensitivity of the data.
        
        Args:
            data: Input data series
            method: Calculation method ("robust", "range", "std")
            
        Returns:
            Calculated sensitivity
        """
        if method == "robust":
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            return 1.5 * iqr if iqr > 0 else data.std()
        elif method == "range":
            return data.max() - data.min()
        elif method == "std":
            return data.std()
        else:
            raise ValueError("Method must be 'robust', 'range', or 'std'")
    
    def get_privacy_budget_used(self) -> float:
        """
        Get total privacy budget used.
        
        Returns:
            Total epsilon used
        """
        return self.epsilon_used
    
    def reset_privacy_budget(self) -> None:
        """Reset privacy budget tracking."""
        self.epsilon_used = 0.0
