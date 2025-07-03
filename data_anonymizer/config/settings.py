"""
Configuration module for the Data Anonymization Tool.
Enhanced version of the original config_module.py
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class AnonymizationConfig:
    """Enhanced configuration class for anonymization settings."""
    
    # Default privacy parameters
    DEFAULT_EPSILON = 1.0
    DEFAULT_DELTA = 1e-5
    DEFAULT_K = 2
    DEFAULT_STRATEGY = 'generalization'
    DEFAULT_NOISE_MECHANISM = 'laplace'
    
    # File paths
    DEFAULT_OUTPUT_DIR = Path("data")
    DEFAULT_CONFIG_FILE = Path("config.json")
    
    # Validation settings
    MIN_EPSILON = 0.01
    MAX_EPSILON = 10.0
    MIN_K = 2
    MAX_K = 20
    MIN_DELTA = 1e-10
    MAX_DELTA = 1e-3
    
    # Supported file formats
    SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx', '.parquet']
    
    # Column type detection patterns
    COLUMN_PATTERNS = {
        'zip_code': ['zip', 'postal', 'code', 'zipcode'],
        'gender': ['gender', 'sex'],
        'occupation': ['occupation', 'job', 'profession', 'work', 'career'],
        'location': ['city', 'town', 'state', 'province', 'country', 'location'],
        'education': ['education', 'degree', 'school', 'university', 'college'],
        'age': ['age', 'birth', 'born', 'dob'],
        'income': ['income', 'salary', 'wage', 'pay', 'earnings'],
        'id': ['id', 'identifier', 'ssn', 'social', 'account'],
        'name': ['name', 'first', 'last', 'firstname', 'lastname'],
        'email': ['email', 'mail', '@'],
        'phone': ['phone', 'tel', 'mobile', 'cell'],
        'address': ['address', 'street', 'road', 'avenue']
    }
    
    # Noise mechanisms
    NOISE_MECHANISMS = ['laplace', 'gaussian']
    
    # K-anonymity strategies
    K_ANONYMITY_STRATEGIES = ['suppression', 'generalization', 'synthetic']
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_FILE
        self.config = self.load_from_file(self.config_path)
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not config_path.exists():
            logger.info(f"Config file not found at {config_path}, using defaults")
            return cls.get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error loading config file: {e}, using defaults")
            return cls.get_default_config()
    
    @classmethod
    def save_to_file(cls, config: Dict[str, Any], config_path: Path) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            raise
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'privacy_settings': {
                'epsilon': cls.DEFAULT_EPSILON,
                'delta': cls.DEFAULT_DELTA,
                'k': cls.DEFAULT_K,
                'strategy': cls.DEFAULT_STRATEGY,
                'noise_mechanism': cls.DEFAULT_NOISE_MECHANISM
            },
            'output_settings': {
                'output_dir': str(cls.DEFAULT_OUTPUT_DIR),
                'include_report': True,
                'save_original': False,
                'report_format': 'json'
            },
            'validation_settings': {
                'min_epsilon': cls.MIN_EPSILON,
                'max_epsilon': cls.MAX_EPSILON,
                'min_k': cls.MIN_K,
                'max_k': cls.MAX_K,
                'min_delta': cls.MIN_DELTA,
                'max_delta': cls.MAX_DELTA
            },
            'column_detection': cls.COLUMN_PATTERNS,
            'processing_settings': {
                'chunk_size': 10000,
                'parallel_processing': False,
                'memory_efficient': True
            },
            'logging_settings': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': False,
                'log_file': 'anonymization.log'
            }
        }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate privacy settings
        privacy = config.get('privacy_settings', {})
        epsilon = privacy.get('epsilon', cls.DEFAULT_EPSILON)
        k = privacy.get('k', cls.DEFAULT_K)
        delta = privacy.get('delta', cls.DEFAULT_DELTA)
        strategy = privacy.get('strategy', cls.DEFAULT_STRATEGY)
        noise_mechanism = privacy.get('noise_mechanism', cls.DEFAULT_NOISE_MECHANISM)
        
        if not (cls.MIN_EPSILON <= epsilon <= cls.MAX_EPSILON):
            errors.append(f"Epsilon must be between {cls.MIN_EPSILON} and {cls.MAX_EPSILON}")
        
        if not (cls.MIN_K <= k <= cls.MAX_K):
            errors.append(f"K must be between {cls.MIN_K} and {cls.MAX_K}")
        
        if not (cls.MIN_DELTA <= delta <= cls.MAX_DELTA):
            errors.append(f"Delta must be between {cls.MIN_DELTA} and {cls.MAX_DELTA}")
        
        if strategy not in cls.K_ANONYMITY_STRATEGIES:
            errors.append(f"Strategy must be one of: {cls.K_ANONYMITY_STRATEGIES}")
        
        if noise_mechanism not in cls.NOISE_MECHANISMS:
            errors.append(f"Noise mechanism must be one of: {cls.NOISE_MECHANISMS}")
        
        # Validate processing settings
        processing = config.get('processing_settings', {})
        chunk_size = processing.get('chunk_size', 10000)
        
        if chunk_size < 100 or chunk_size > 1000000:
            errors.append("Chunk size must be between 100 and 1,000,000")
        
        return errors
    
    def get_privacy_settings(self) -> Dict[str, Any]:
        """Get privacy settings from config."""
        return self.config.get('privacy_settings', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings from config."""
        return self.config.get('output_settings', {})
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get validation settings from config."""
        return self.config.get('validation_settings', {})
    
    def get_column_patterns(self) -> Dict[str, List[str]]:
        """Get column detection patterns."""
        return self.config.get('column_detection', self.COLUMN_PATTERNS)
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """Get processing settings from config."""
        return self.config.get('processing_settings', {})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        def deep_update(original: Dict, updates: Dict) -> Dict:
            """Recursively update nested dictionaries."""
            for key, value in updates.items():
                if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
            return original
        
        deep_update(self.config, updates)
        
        # Validate updated config
        errors = self.validate_config(self.config)
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
    
    def save_config(self, config_path: Optional[Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save config (uses default if not provided)
        """
        save_path = config_path or self.config_path
        self.save_to_file(self.config, save_path)
    
    def detect_column_types(self, columns: List[str]) -> Dict[str, str]:
        """
        Detect column types based on patterns.
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to detected types
        """
        detected = {}
        patterns = self.get_column_patterns()
        
        for col in columns:
            col_lower = col.lower()
            detected_type = 'unknown'
            
            for type_name, keywords in patterns.items():
                if any(keyword in col_lower for keyword in keywords):
                    detected_type = type_name
                    break
            
            detected[col] = detected_type
        
        return detected

# Example configuration templates
PRIVACY_TEMPLATES = {
    'high_privacy': {
        'epsilon': 0.1,
        'delta': 1e-6,
        'k': 5,
        'strategy': 'synthetic',
        'noise_mechanism': 'gaussian'
    },
    'medium_privacy': {
        'epsilon': 1.0,
        'delta': 1e-5,
        'k': 3,
        'strategy': 'generalization',
        'noise_mechanism': 'laplace'
    },
    'low_privacy': {
        'epsilon': 2.0,
        'delta': 1e-4,
        'k': 2,
        'strategy': 'suppression',
        'noise_mechanism': 'laplace'
    },
    'research_compliant': {
        'epsilon': 0.5,
        'delta': 1e-5,
        'k': 3,
        'strategy': 'generalization',
        'noise_mechanism': 'gaussian'
    },
    'minimal_anonymization': {
        'epsilon': 5.0,
        'delta': 1e-3,
        'k': 2,
        'strategy': 'suppression',
        'noise_mechanism': 'laplace'
    }
}

# Configuration helpers
def create_config_from_template(template_name: str, 
                              custom_settings: Optional[Dict[str, Any]] = None) -> AnonymizationConfig:
    """
    Create configuration from template.
    
    Args:
        template_name: Name of the privacy template
        custom_settings: Optional custom settings to override
        
    Returns:
        AnonymizationConfig instance
    """
    if template_name not in PRIVACY_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(PRIVACY_TEMPLATES.keys())}")
    
    config = AnonymizationConfig.get_default_config()
    config['privacy_settings'].update(PRIVACY_TEMPLATES[template_name])
    
    if custom_settings:
        def deep_update(original: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                    deep_update(original[key], value)
                else:
                    original[key] = value
            return original
        
        deep_update(config, custom_settings)
    
    # Create config instance
    config_instance = AnonymizationConfig()
    config_instance.config = config
    
    return config_instance
