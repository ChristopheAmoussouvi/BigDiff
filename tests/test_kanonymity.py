"""
Tests for the K-Anonymity module.
"""

import pytest
import pandas as pd
from data_anonymizer.core.kanonymity import KAnonymity

@pytest.fixture
def k_anon_engine():
    """Fixture for KAnonymity engine."""
    return KAnonymity(random_seed=42)

@pytest.fixture
def sample_data():
    """Fixture for sample categorical data."""
    data = {
        'zipcode': ['12345', '12345', '54321', '54321', '54321'],
        'gender': ['M', 'F', 'M', 'F', 'F'],
        'age': [25, 30, 35, 40, 45]
    }
    return pd.DataFrame(data)

def test_apply_k_anonymity_suppression(k_anon_engine: KAnonymity, sample_data: pd.DataFrame):
    """Test k-anonymity with suppression strategy."""
    anonymized_df, stats = k_anon_engine.apply_k_anonymity(
        sample_data, quasi_identifiers=['zipcode', 'gender'], k=2, strategy='suppression'
    )
    
    assert stats['records_modified'] == 2
    assert anonymized_df.loc[0, 'zipcode'] == '*****'
    assert anonymized_df.loc[1, 'gender'] == '*'

def test_apply_k_anonymity_generalization(k_anon_engine: KAnonymity, sample_data: pd.DataFrame):
    """Test k-anonymity with generalization strategy."""
    anonymized_df, stats = k_anon_engine.apply_k_anonymity(
        sample_data, quasi_identifiers=['zipcode'], k=3, strategy='generalization'
    )
    
    assert stats['records_modified'] == 2
    assert anonymized_df.loc[0, 'zipcode'] == '12***'

def test_apply_k_anonymity_synthetic(k_anon_engine: KAnonymity, sample_data: pd.DataFrame):
    """Test k-anonymity with synthetic strategy."""
    anonymized_df, stats = k_anon_engine.apply_k_anonymity(
        sample_data, quasi_identifiers=['zipcode'], k=3, strategy='synthetic'
    )
    
    assert stats['records_modified'] == 2
    assert anonymized_df.loc[0, 'zipcode'] != '12345'

def test_validate_k_anonymity(k_anon_engine: KAnonymity, sample_data: pd.DataFrame):
    """Test k-anonymity validation."""
    validation = k_anon_engine.validate_k_anonymity(sample_data, ['zipcode'], k=2)
    assert not validation['k_anonymity_satisfied']
    
    validation = k_anon_engine.validate_k_anonymity(sample_data, ['gender'], k=2)
    assert validation['k_anonymity_satisfied']

def test_information_loss(k_anon_engine: KAnonymity, sample_data: pd.DataFrame):
    """Test information loss calculation."""
    anonymized_df, _ = k_anon_engine.apply_k_anonymity(
        sample_data, quasi_identifiers=['zipcode'], k=3, strategy='suppression'
    )
    
    loss_metrics = k_anon_engine.calculate_information_loss(
        sample_data, anonymized_df, ['zipcode']
    )
    
    assert 'zipcode' in loss_metrics
    zipcode_metrics = loss_metrics.get('zipcode')
    if isinstance(zipcode_metrics, dict):
        assert zipcode_metrics['uniqueness_loss'] > 0
    else:
        pytest.fail("zipcode_metrics is not a dictionary")
