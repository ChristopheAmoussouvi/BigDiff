"""
Tests for the main DataAnonymizer class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from data_anonymizer.core.anonymizer import DataAnonymizer
from data_anonymizer.config.settings import AnonymizationConfig

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture for a sample DataFrame."""
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'zipcode': ['12345', '12345', '54321', '54321', '54321'],
        'salary': [50000, 60000, 70000, 80000, 90000]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Fixture to create a temporary CSV file."""
    file_path = tmp_path / "sample.csv"
    sample_df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def anonymizer() -> DataAnonymizer:
    """Fixture for a DataAnonymizer instance."""
    return DataAnonymizer(random_seed=42)

def test_load_data(anonymizer: DataAnonymizer, temp_csv_file: Path):
    """Test loading data from a CSV file."""
    df = anonymizer.load_data(temp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert anonymizer.original_df is not None

def test_apply_differential_privacy(anonymizer: DataAnonymizer, sample_df: pd.DataFrame):
    """Test applying differential privacy to numerical columns."""
    anonymizer.df = sample_df.copy()
    anonymizer.original_df = sample_df.copy()
    
    anonymized_df = anonymizer.apply_differential_privacy(['age', 'salary'])
    
    assert not anonymized_df['age'].equals(sample_df['age'])
    assert not anonymized_df['salary'].equals(sample_df['salary'])

def test_apply_k_anonymity(anonymizer: DataAnonymizer, sample_df: pd.DataFrame):
    """Test applying k-anonymity to categorical columns."""
    anonymizer.df = sample_df.copy()
    anonymizer.original_df = sample_df.copy()
    
    anonymized_df = anonymizer.apply_k_anonymity(['zipcode'], k=2)
    
    assert not anonymized_df['zipcode'].equals(sample_df['zipcode'])

def test_save_anonymized_data(anonymizer: DataAnonymizer, sample_df: pd.DataFrame, tmp_path: Path):
    """Test saving anonymized data and report."""
    anonymizer.df = sample_df.copy()
    output_path = tmp_path / "anonymized.csv"
    
    anonymizer.save_anonymized_data(output_path, include_report=True)
    
    assert output_path.exists()
    report_path = tmp_path / "anonymized_report.json"
    assert report_path.exists()

def test_get_anonymization_report(anonymizer: DataAnonymizer, sample_df: pd.DataFrame):
    """Test generating an anonymization report."""
    anonymizer.df = sample_df.copy()
    anonymizer.original_df = sample_df.copy()
    
    anonymizer.apply_differential_privacy(['age'])
    report = anonymizer.get_anonymization_report()
    
    assert 'metadata' in report
    assert 'utility_metrics' in report
    assert len(report['anonymization_log']) == 1

def test_reset(anonymizer: DataAnonymizer, sample_df: pd.DataFrame):
    """Test resetting the anonymizer."""
    anonymizer.df = sample_df.copy()
    anonymizer.original_df = sample_df.copy()
    
    anonymizer.apply_differential_privacy(['age'])
    anonymizer.reset()
    
    assert anonymizer.df.equals(anonymizer.original_df)
    assert len(anonymizer.anonymization_log) == 0
    assert anonymizer.dp_engine.get_privacy_budget_used() == 0.0

def test_get_column_recommendations(anonymizer: DataAnonymizer, sample_df: pd.DataFrame):
    """Test column recommendations."""
    anonymizer.df = sample_df.copy()
    recommendations = anonymizer.get_column_recommendations()
    
    assert 'age' in recommendations['numerical_columns']
    assert 'zipcode' in recommendations['quasi_identifiers']
    assert 'name' in recommendations['sensitive_attributes']
