"""
Data Anonymizer Tool
"""

__version__ = "1.0.0"

from .core.anonymizer import DataAnonymizer
from .config.settings import AnonymizationConfig, create_config_from_template

__all__ = ["DataAnonymizer", "AnonymizationConfig", "create_config_from_template"]
