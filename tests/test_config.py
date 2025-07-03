"""
Tests for the configuration module.
"""

import pytest
import json
from pathlib import Path
from data_anonymizer.config.settings import AnonymizationConfig, PRIVACY_TEMPLATES, create_config_from_template

@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Fixture to create a temporary config file."""
    config_data = {
        "privacy_settings": {
            "epsilon": 2.0,
            "k": 3
        }
    }
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path

def test_load_from_file(temp_config_file: Path):
    """Test loading configuration from a file."""
    config = AnonymizationConfig.load_from_file(temp_config_file)
    assert config['privacy_settings']['epsilon'] == 2.0
    assert config['privacy_settings']['k'] == 3

def test_load_from_nonexistent_file():
    """Test loading from a non-existent file returns defaults."""
    config = AnonymizationConfig.load_from_file(Path("nonexistent.json"))
    default_config = AnonymizationConfig.get_default_config()
    assert config == default_config

def test_save_to_file(tmp_path: Path):
    """Test saving configuration to a file."""
    config_data = AnonymizationConfig.get_default_config()
    config_path = tmp_path / "new_config.json"
    AnonymizationConfig.save_to_file(config_data, config_path)
    
    assert config_path.exists()
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    assert saved_config == config_data

def test_validate_config():
    """Test configuration validation."""
    valid_config = AnonymizationConfig.get_default_config()
    assert AnonymizationConfig.validate_config(valid_config) == []
    
    invalid_config = {
        "privacy_settings": {
            "epsilon": 100,
            "k": 1,
            "strategy": "invalid_strategy"
        }
    }
    errors = AnonymizationConfig.validate_config(invalid_config)
    assert len(errors) == 3

def test_create_config_from_template():
    """Test creating configuration from a template."""
    config = create_config_from_template('high_privacy')
    assert config.get_privacy_settings()['epsilon'] == 0.1
    assert config.get_privacy_settings()['k'] == 5
    
    with pytest.raises(ValueError):
        create_config_from_template('nonexistent_template')

def test_update_config():
    """Test updating configuration."""
    config = AnonymizationConfig()
    updates = {
        "privacy_settings": {
            "epsilon": 5.0
        },
        "output_settings": {
            "include_report": False
        }
    }
    config.update_config(updates)
    
    assert config.get_privacy_settings()['epsilon'] == 5.0
    assert not config.get_output_settings()['include_report']

def test_detect_column_types():
    """Test column type detection."""
    config = AnonymizationConfig()
    columns = ['age', 'zipcode', 'first_name', 'salary', 'job_title']
    detected = config.detect_column_types(columns)
    
    assert detected['age'] == 'age'
    assert detected['zipcode'] == 'zip_code'
    assert detected['first_name'] == 'name'
    assert detected['salary'] == 'income'
    assert detected['job_title'] == 'occupation'
