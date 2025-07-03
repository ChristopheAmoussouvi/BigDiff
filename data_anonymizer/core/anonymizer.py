"""
Enhanced Data Anonymizer - Main anonymization class.
Refactored and improved version of enhanced_anonymizer.py
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import json
import warnings

from .privacy import DifferentialPrivacy
from .kanonymity import KAnonymity
from ..config.settings import AnonymizationConfig

logger = logging.getLogger(__name__)

class DataAnonymizer:
    """
    Enhanced data anonymization tool using Differential Privacy and K-Anonymity.
    
    This class provides robust anonymization techniques for sensitive datasets:
    - Differential Privacy for numerical columns with proper mathematical implementation
    - K-Anonymity strategies for categorical columns (suppression, generalization, synthetic)
    - Comprehensive validation and error handling
    - Detailed anonymization reports and metrics
    - Configurable privacy parameters
    """

    def __init__(self, 
                 config: Optional[AnonymizationConfig] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the DataAnonymizer.
        
        Args:
            config: Optional configuration object
            random_seed: Optional seed for reproducible results
        """
        self.config = config or AnonymizationConfig()
        self.random_seed = random_seed
        
        # Initialize privacy mechanisms
        self.dp_engine = DifferentialPrivacy(random_seed=random_seed)
        self.k_anon_engine = KAnonymity(random_seed=random_seed)
        
        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.anonymization_log: List[Dict] = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info("DataAnonymizer initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_settings = self.config.get_processing_settings().get('logging_settings', {})
        level = getattr(logging, log_settings.get('level', 'INFO'))
        
        logging.basicConfig(
            level=level,
            format=log_settings.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        )

    def load_data(self, 
                  filepath: Union[str, Path],
                  file_format: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load dataset from various file formats with validation.
        
        Args:
            filepath: Path to the data file
            file_format: Optional file format override
            **kwargs: Additional arguments for pandas readers
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid format
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File '{filepath}' not found.")
        
        # Determine file format
        if file_format is None:
            file_format = filepath.suffix.lower()
        
        if file_format not in self.config.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_format}. "
                           f"Supported formats: {self.config.SUPPORTED_FORMATS}")
        
        try:
            # Load based on file format
            if file_format == '.csv':
                self.df = pd.read_csv(filepath, **kwargs)
            elif file_format == '.json':
                self.df = pd.read_json(filepath, **kwargs)
            elif file_format == '.xlsx':
                self.df = pd.read_excel(filepath, **kwargs)
            elif file_format == '.parquet':
                self.df = pd.read_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            if self.df is not None:
                self.original_df = self.df.copy()
            
        except Exception as e:
            raise ValueError(f"Error reading {file_format} file: {e}")
        
        if self.df is None or self.df.empty:
            raise ValueError("Dataset is empty.")
        
        logger.info(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Auto-detect column types
        detected_types = self.config.detect_column_types(list(self.df.columns))
        logger.info(f"Auto-detected column types: {detected_types}")
        
        return self.df

    def validate_columns(self, columns: List[str], column_type: str = "specified") -> List[str]:
        """
        Validate that specified columns exist in the dataset.
        
        Args:
            columns: List of column names to validate
            column_type: Type description for error messages
            
        Returns:
            List of valid column names
        """
        if not columns or (len(columns) == 1 and columns[0] == ''):
            return []
        
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        
        valid_columns = []
        for col in columns:
            if col in self.df.columns:
                valid_columns.append(col)
            else:
                logger.warning(f"{column_type} column '{col}' not found in dataset")
        
        return valid_columns

    def apply_differential_privacy(self, 
                                 numerical_columns: List[str], 
                                 epsilon: Optional[float] = None,
                                 delta: Optional[float] = None,
                                 noise_mechanism: str = 'laplace',
                                 clipping_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """
        Apply Differential Privacy to numerical columns.
        
        Args:
            numerical_columns: List of numerical column names
            epsilon: Privacy budget (uses config default if None)
            delta: Failure probability (uses config default if None)
            noise_mechanism: Noise mechanism ('laplace' or 'gaussian')
            clipping_bounds: Optional bounds for clipping values {col: (min, max)}
            
        Returns:
            DataFrame with differential privacy applied
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        
        # Use config defaults if not specified
        privacy_settings = self.config.get_privacy_settings()
        epsilon_val = epsilon if epsilon is not None else privacy_settings.get('epsilon', 1.0)
        delta_val = delta if delta is not None else privacy_settings.get('delta', 1e-5)
        noise_mechanism = noise_mechanism or privacy_settings.get('noise_mechanism', 'laplace')
        
        if epsilon_val <= 0:
            raise ValueError("Epsilon must be positive")
        
        valid_columns = self.validate_columns(numerical_columns, "numerical")
        
        for col in valid_columns:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping")
                continue
            
            # Apply noise based on mechanism
            original_values = self.df[col].copy()
            clipping_bound = clipping_bounds.get(col) if clipping_bounds else None
            
            if noise_mechanism == 'laplace':
                self.df[col] = self.dp_engine.apply_laplace_noise(
                    self.df[col], epsilon_val, clipping_bounds=clipping_bound
                )
            elif noise_mechanism == 'gaussian':
                self.df[col] = self.dp_engine.apply_gaussian_noise(
                    self.df[col], epsilon_val, delta_val, clipping_bounds=clipping_bound
                )
            else:
                raise ValueError(f"Unknown noise mechanism: {noise_mechanism}")
            
            # Round to reasonable precision
            if original_values.dtype in ['int64', 'int32']:
                self.df[col] = self.df[col].round().astype(int)
            else:
                self.df[col] = self.df[col].round(2)
            
            # Log the anonymization
            self.anonymization_log.append({
                'column': col,
                'method': 'differential_privacy',
                'mechanism': noise_mechanism,
                'epsilon': epsilon_val,
                'delta': delta_val if noise_mechanism == 'gaussian' else None,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Differential Privacy applied to '{col}' (ε={epsilon_val}, mechanism={noise_mechanism})")
        
        return self.df

    def apply_k_anonymity(self, 
                         quasi_identifiers: List[str], 
                         k: Optional[int] = None, 
                         strategy: Optional[str] = None) -> pd.DataFrame:
        """
        Apply K-Anonymity to categorical columns.
        
        Args:
            quasi_identifiers: List of quasi-identifier column names
            k: Minimum group size for k-anonymity (uses config default if None)
            strategy: Strategy ('suppression', 'generalization', 'synthetic')
            
        Returns:
            DataFrame with k-anonymity applied
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call load_data() first.")
        
        # Use config defaults if not specified
        privacy_settings = self.config.get_privacy_settings()
        k_val = k if k is not None else privacy_settings.get('k', 2)
        strategy_val = strategy if strategy is not None else privacy_settings.get('strategy', 'generalization')
        
        valid_columns = self.validate_columns(quasi_identifiers, "quasi-identifier")
        
        if not valid_columns:
            logger.warning("No valid quasi-identifier columns found")
            return self.df
        
        # Apply k-anonymity
        anonymized_df, stats = self.k_anon_engine.apply_k_anonymity(
            self.df, valid_columns, k_val, strategy_val
        )
        
        self.df = anonymized_df
        
        # Log the anonymization
        self.anonymization_log.append({
            'columns': valid_columns,
            'method': 'k_anonymity',
            'k': k_val,
            'strategy': strategy_val,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"K-Anonymity applied with k={k_val}, strategy={strategy_val}")
        logger.info(f"Modified {stats['records_modified']} records")
        
        return self.df

    def save_anonymized_data(self, 
                           output_path: Union[str, Path],
                           file_format: Optional[str] = None,
                           include_report: Optional[bool] = None) -> Path:
        """
        Save anonymized data with optional anonymization report.
        
        Args:
            output_path: Output file path
            file_format: Optional file format override
            include_report: Whether to save anonymization report
            
        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No dataset to save. Apply anonymization first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format
        if file_format is None:
            file_format = output_path.suffix.lower()
        
        # Save based on format
        try:
            if file_format == '.csv':
                self.df.to_csv(output_path, index=False)
            elif file_format == '.json':
                self.df.to_json(output_path, orient='records', indent=2)
            elif file_format == '.xlsx':
                self.df.to_excel(output_path, index=False)
            elif file_format == '.parquet':
                self.df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {file_format}")
                
            logger.info(f"Anonymized dataset saved to '{output_path}'")
            
        except Exception as e:
            raise ValueError(f"Error saving file: {e}")
        
        # Save anonymization report if requested
        output_settings = self.config.get_output_settings()
        if include_report or (include_report is None and output_settings.get('include_report', True)):
            report_path = output_path.parent / f"{output_path.stem}_report.json"
            report_data = self._generate_full_report()
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Anonymization report saved to '{report_path}'")
        
        return output_path

    def _generate_full_report(self) -> Dict[str, Any]:
        """Generate comprehensive anonymization report."""
        if self.original_df is None or self.df is None:
            raise ValueError("No data available for report generation")
        
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'anonymizer_version': '1.0.0',
                'config_used': self.config.config
            },
            'dataset_info': {
                'original_shape': self.original_df.shape,
                'anonymized_shape': self.df.shape,
                'columns': list(self.df.columns)
            },
            'anonymization_log': self.anonymization_log,
            'privacy_analysis': self._generate_privacy_analysis(),
            'utility_metrics': self._generate_utility_metrics(),
            'validation_results': self._generate_validation_results()
        }

    def _generate_privacy_analysis(self) -> Dict[str, Any]:
        """Generate privacy analysis metrics."""
        analysis = {
            'total_epsilon_used': self.dp_engine.get_privacy_budget_used(),
            'differential_privacy_applied': any(log['method'] == 'differential_privacy' for log in self.anonymization_log),
            'k_anonymity_applied': any(log['method'] == 'k_anonymity' for log in self.anonymization_log)
        }
        
        # K-anonymity validation
        k_anon_logs = [log for log in self.anonymization_log if log['method'] == 'k_anonymity']
        if k_anon_logs and self.df is not None:
            latest_k_anon = k_anon_logs[-1]
            validation = self.k_anon_engine.validate_k_anonymity(
                self.df, latest_k_anon['columns'], latest_k_anon['k']
            )
            analysis['k_anonymity_validation'] = validation
        
        return analysis

    def _generate_utility_metrics(self) -> Dict[str, Any]:
        """Generate data utility metrics."""
        if self.original_df is None or self.df is None:
            return {}
        
        metrics = {}
        
        for col in self.original_df.columns:
            if col in self.df.columns:
                original_unique = self.original_df[col].nunique()
                anonymized_unique = self.df[col].nunique()
                
                metrics[col] = {
                    'original_unique_values': original_unique,
                    'anonymized_unique_values': anonymized_unique,
                    'uniqueness_retention': anonymized_unique / original_unique if original_unique > 0 else 0,
                    'data_type': str(self.df[col].dtype)
                }
                
                # Additional metrics for numerical columns
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    try:
                        metrics[col].update({
                            'original_mean': float(self.original_df[col].mean()),
                            'anonymized_mean': float(self.df[col].mean()),
                            'original_std': float(self.original_df[col].std()),
                            'anonymized_std': float(self.df[col].std()),
                        })
                    except:
                        pass  # Skip if calculation fails
        
        return metrics

    def _generate_validation_results(self) -> Dict[str, Any]:
        """Generate validation results."""
        if self.original_df is None or self.df is None:
            return {}
            
        results = {
            'data_integrity': {
                'shape_preserved': self.original_df.shape == self.df.shape,
                'columns_preserved': list(self.original_df.columns) == list(self.df.columns),
                'no_null_introduced': not self.df.isnull().any().any() or self.original_df.isnull().any().any()
            }
        }
        
        return results

    def get_anonymization_report(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Generate and display anonymization report.
        
        Args:
            detailed: Whether to return detailed report
            
        Returns:
            Dictionary containing anonymization metrics
        """
        if self.original_df is None or self.df is None:
            raise ValueError("No data loaded for comparison")
        
        report = self._generate_full_report() if detailed else {
            'dataset_info': {
                'original_shape': self.original_df.shape,
                'anonymized_shape': self.df.shape,
                'columns': list(self.df.columns)
            },
            'anonymization_methods': self.anonymization_log,
            'utility_metrics': self._generate_utility_metrics()
        }
        
        # Print formatted report
        self._print_report_summary(report)
        
        return report

    def _print_report_summary(self, report: Dict[str, Any]) -> None:
        """Print formatted report summary."""
        print("\n" + "="*80)
        print("ANONYMIZATION REPORT")
        print("="*80)
        
        dataset_info = report.get('dataset_info', {})
        print(f"Dataset: {dataset_info.get('original_shape', ['?', '?'])[0]} rows, "
              f"{dataset_info.get('original_shape', ['?', '?'])[1]} columns")
        print(f"Anonymization methods applied: {len(self.anonymization_log)}")
        
        # Privacy budget summary
        if 'privacy_analysis' in report:
            privacy = report['privacy_analysis']
            print(f"Total privacy budget used (ε): {privacy.get('total_epsilon_used', 0):.3f}")
        
        # Utility metrics table
        utility_metrics = report.get('utility_metrics', {})
        if utility_metrics:
            print(f"\n{'Column':<20} {'Original':<15} {'Anonymized':<15} {'Retention %':<15}")
            print("-" * 80)
            
            for col, stats in utility_metrics.items():
                retention_pct = stats.get('uniqueness_retention', 0) * 100
                print(f"{col:<20} {stats.get('original_unique_values', 0):<15} "
                      f"{stats.get('anonymized_unique_values', 0):<15} {retention_pct:<14.1f}%")
        
        print("-" * 80)

    def validate_anonymization(self, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate that anonymization requirements are met.
        
        Args:
            k: Minimum group size to validate (uses config default if None)
            
        Returns:
            Validation results
        """
        if self.df is None:
            raise ValueError("No dataset loaded for validation")
        
        validation_results = {}
        
        # Check if we have quasi-identifiers from the log
        k_anon_logs = [log for log in self.anonymization_log if log['method'] == 'k_anonymity']
        
        if k_anon_logs:
            latest_log = k_anon_logs[-1]
            k_value = k or latest_log.get('k', 2)
            quasi_identifiers = latest_log.get('columns', [])
            
            if quasi_identifiers:
                validation_results = self.k_anon_engine.validate_k_anonymity(
                    self.df, quasi_identifiers, k_value
                )
        
        # Add privacy budget validation
        privacy_settings = self.config.get_privacy_settings()
        max_epsilon = self.config.get_validation_settings().get('max_epsilon', 10.0)
        
        validation_results.update({
            'privacy_budget_within_limits': self.dp_engine.get_privacy_budget_used() <= max_epsilon,
            'total_epsilon_used': self.dp_engine.get_privacy_budget_used(),
            'epsilon_limit': max_epsilon
        })
        
        return validation_results

    def reset(self) -> None:
        """Reset the anonymizer to initial state."""
        self.df = self.original_df.copy() if self.original_df is not None else None
        self.anonymization_log = []
        self.dp_engine.reset_privacy_budget()
        logger.info("Anonymizer reset to initial state")

    def get_column_recommendations(self) -> Dict[str, List[str]]:
        """
        Get column recommendations based on detected types.
        
        Returns:
            Dictionary with recommended columns for different anonymization types
        """
        if self.df is None:
            raise ValueError("No dataset loaded")
        
        detected_types = self.config.detect_column_types(list(self.df.columns))
        
        recommendations = {
            'numerical_columns': [],
            'quasi_identifiers': [],
            'sensitive_attributes': []
        }
        
        for col, col_type in detected_types.items():
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if col_type in ['age', 'income']:
                    recommendations['numerical_columns'].append(col)
            
            if col_type in ['zip_code', 'occupation', 'education', 'location']:
                recommendations['quasi_identifiers'].append(col)
            
            if col_type in ['id', 'name', 'email', 'phone', 'address']:
                recommendations['sensitive_attributes'].append(col)
        
        return recommendations
