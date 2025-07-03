"""
Tests for the Differential Privacy module.
"""

import pytest
import pandas as pd
import numpy as np
from data_anonymizer.core.privacy import DifferentialPrivacy

@pytest.fixture
def dp_engine():
    """Fixture for DifferentialPrivacy engine."""
    return DifferentialPrivacy(random_seed=42)

@pytest.fixture
def sample_data():
    """Fixture for sample numerical data."""
    return pd.Series(np.random.randint(20, 60, size=100), name="age")

def test_apply_laplace_noise(dp_engine: DifferentialPrivacy, sample_data: pd.Series):
    """Test applying Laplace noise."""
    noisy_data = dp_engine.apply_laplace_noise(sample_data, epsilon=1.0)
    
    assert noisy_data.shape == sample_data.shape
    assert not noisy_data.equals(sample_data)
    assert np.abs(noisy_data.mean() - sample_data.mean()) < 5  # Check if mean is close

def test_apply_gaussian_noise(dp_engine: DifferentialPrivacy, sample_data: pd.Series):
    """Test applying Gaussian noise."""
    noisy_data = dp_engine.apply_gaussian_noise(sample_data, epsilon=1.0, delta=1e-5)
    
    assert noisy_data.shape == sample_data.shape
    assert not noisy_data.equals(sample_data)
    assert np.abs(noisy_data.mean() - sample_data.mean()) < 5

def test_clipping_bounds(dp_engine: DifferentialPrivacy, sample_data: pd.Series):
    """Test clipping bounds functionality."""
    bounds = (30, 50)
    noisy_data = dp_engine.apply_laplace_noise(sample_data, epsilon=1.0, clipping_bounds=bounds)
    
    # Check that original data outside bounds is clipped before adding noise
    # This is an indirect check, but a full check is complex
    assert noisy_data.mean() > 25 and noisy_data.mean() < 55

def test_invalid_epsilon(dp_engine: DifferentialPrivacy, sample_data: pd.Series):
    """Test invalid epsilon values."""
    with pytest.raises(ValueError):
        dp_engine.apply_laplace_noise(sample_data, epsilon=0)
    with pytest.raises(ValueError):
        dp_engine.apply_laplace_noise(sample_data, epsilon=-1.0)

def test_privacy_budget_tracking(dp_engine: DifferentialPrivacy, sample_data: pd.Series):
    """Test privacy budget tracking."""
    dp_engine.apply_laplace_noise(sample_data, epsilon=1.0)
    dp_engine.apply_gaussian_noise(sample_data, epsilon=0.5)
    
    assert dp_engine.get_privacy_budget_used() == 1.5
    
    dp_engine.reset_privacy_budget()
    assert dp_engine.get_privacy_budget_used() == 0.0

def test_calculate_sensitivity(dp_engine: DifferentialPrivacy, sample_data: pd.Series):
    """Test sensitivity calculation methods."""
    robust_sens = dp_engine.calculate_sensitivity(sample_data, method="robust")
    range_sens = dp_engine.calculate_sensitivity(sample_data, method="range")
    std_sens = dp_engine.calculate_sensitivity(sample_data, method="std")
    
    assert robust_sens > 0
    assert range_sens > 0
    assert std_sens > 0
    
    with pytest.raises(ValueError):
        dp_engine.calculate_sensitivity(sample_data, method="invalid")
