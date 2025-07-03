"""
K-Anonymity implementation for categorical data anonymization.
"""

import pandas as pd
import numpy as np
import random
from faker import Faker
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class KAnonymity:
    """
    K-Anonymity implementation with multiple strategies.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize KAnonymity.
        
        Args:
            random_seed: Optional seed for reproducible results
        """
        self.fake = Faker()
        if random_seed:
            Faker.seed(random_seed)
            random.seed(random_seed)
    
    def apply_k_anonymity(self, 
                         data: pd.DataFrame,
                         quasi_identifiers: List[str], 
                         k: int = 2, 
                         strategy: str = 'generalization') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply K-Anonymity to categorical columns.
        
        Args:
            data: Input DataFrame
            quasi_identifiers: List of quasi-identifier column names
            k: Minimum group size for k-anonymity
            strategy: Strategy ('suppression', 'generalization', 'synthetic')
            
        Returns:
            Tuple of (anonymized DataFrame, statistics)
        """
        if k < 2:
            raise ValueError("k must be at least 2")
        
        valid_strategies = ['suppression', 'generalization', 'synthetic']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of: {valid_strategies}")
        
        # Validate columns exist
        missing_cols = [col for col in quasi_identifiers if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        df = data.copy()
        
        # Convert to string for consistency
        for col in quasi_identifiers:
            df[col] = df[col].astype(str)
        
        # Group by quasi-identifiers
        groups = df.groupby(quasi_identifiers, dropna=False)
        records_modified = 0
        total_groups = len(groups)
        violating_groups = 0
        
        for group_values, group_indices in groups:
            if len(group_indices) < k:
                violating_groups += 1
                records_modified += len(group_indices)
                
                for i, col in enumerate(quasi_identifiers):
                    if strategy == 'suppression':
                        # Replace with asterisks maintaining length
                        if len(quasi_identifiers) > 1:
                            original_value = str(group_values[i])
                        else:
                            original_value = str(group_values)
                        df.loc[group_indices.index, col] = '*' * len(original_value)
                    
                    elif strategy == 'generalization':
                        # Keep first 2 characters, replace rest with asterisks
                        if len(quasi_identifiers) > 1:
                            original_value = str(group_values[i])
                        else:
                            original_value = str(group_values)
                        if len(original_value) > 2:
                            df.loc[group_indices.index, col] = original_value[:2] + '*' * (len(original_value) - 2)
                        else:
                            df.loc[group_indices.index, col] = '*' * len(original_value)
                    
                    elif strategy == 'synthetic':
                        # Generate synthetic data based on column semantics
                        synthetic_values = self._generate_synthetic_values(col, len(group_indices))
                        df.loc[group_indices.index, col] = synthetic_values
        
        # Generate statistics
        stats = {
            'k': k,
            'strategy': strategy,
            'quasi_identifiers': quasi_identifiers,
            'records_modified': records_modified,
            'total_groups': total_groups,
            'violating_groups': violating_groups,
            'k_anonymity_satisfied': violating_groups == 0,
            'modification_rate': records_modified / len(df) if len(df) > 0 else 0
        }
        
        logger.info(f"K-Anonymity applied with k={k}, strategy={strategy}")
        logger.info(f"Modified {records_modified} records out of {len(df)}")
        
        return df, stats
    
    def _generate_synthetic_values(self, column: str, count: int) -> List[str]:
        """
        Generate synthetic values based on column semantics.
        
        Args:
            column: Column name
            count: Number of values to generate
            
        Returns:
            List of synthetic values
        """
        column_lower = column.lower()
        
        if any(term in column_lower for term in ['zip', 'postal', 'code']):
            return [self.fake.zipcode() for _ in range(count)]
        elif any(term in column_lower for term in ['gender', 'sex']):
            return [random.choice(['Male', 'Female', 'Other']) for _ in range(count)]
        elif any(term in column_lower for term in ['occupation', 'job', 'profession']):
            return [self.fake.job() for _ in range(count)]
        elif any(term in column_lower for term in ['city', 'town']):
            return [self.fake.city() for _ in range(count)]
        elif any(term in column_lower for term in ['state', 'province']):
            return [self.fake.state() for _ in range(count)]
        elif any(term in column_lower for term in ['country', 'nation']):
            return [self.fake.country() for _ in range(count)]
        elif any(term in column_lower for term in ['education', 'degree']):
            education_levels = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD', 'Associate']
            return [random.choice(education_levels) for _ in range(count)]
        elif any(term in column_lower for term in ['age']):
            return [str(random.randint(18, 80)) for _ in range(count)]
        elif any(term in column_lower for term in ['name', 'first', 'last']):
            return [self.fake.name() for _ in range(count)]
        else:
            return [self.fake.word() for _ in range(count)]
    
    def validate_k_anonymity(self, 
                            data: pd.DataFrame, 
                            quasi_identifiers: List[str], 
                            k: int = 2) -> Dict[str, Any]:
        """
        Validate that k-anonymity requirements are met.
        
        Args:
            data: DataFrame to validate
            quasi_identifiers: List of quasi-identifier columns
            k: Minimum group size to validate
            
        Returns:
            Validation results
        """
        # Validate columns exist
        missing_cols = [col for col in quasi_identifiers if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        
        groups = data.groupby(quasi_identifiers, dropna=False)
        group_sizes = [len(group) for _, group in groups]
        
        validation_results = {
            'k_anonymity_satisfied': all(size >= k for size in group_sizes),
            'min_group_size': min(group_sizes) if group_sizes else 0,
            'max_group_size': max(group_sizes) if group_sizes else 0,
            'avg_group_size': np.mean(group_sizes) if group_sizes else 0,
            'violating_groups': sum(1 for size in group_sizes if size < k),
            'total_groups': len(group_sizes),
            'group_size_distribution': {
                'sizes': group_sizes,
                'unique_sizes': list(set(group_sizes))
            }
        }
        
        return validation_results
    
    def calculate_information_loss(self, 
                                  original_data: pd.DataFrame,
                                  anonymized_data: pd.DataFrame,
                                  quasi_identifiers: List[str]) -> Dict[str, float]:
        """
        Calculate information loss metrics.
        
        Args:
            original_data: Original DataFrame
            anonymized_data: Anonymized DataFrame
            quasi_identifiers: List of quasi-identifier columns
            
        Returns:
            Information loss metrics
        """
        metrics = {}
        
        for col in quasi_identifiers:
            if col in original_data.columns and col in anonymized_data.columns:
                original_unique = original_data[col].nunique()
                anonymized_unique = anonymized_data[col].nunique()
                
                # Calculate various metrics
                metrics[col] = {
                    'original_unique': original_unique,
                    'anonymized_unique': anonymized_unique,
                    'uniqueness_loss': (original_unique - anonymized_unique) / original_unique if original_unique > 0 else 0,
                    'retention_rate': anonymized_unique / original_unique if original_unique > 0 else 0
                }
        
        # Overall metrics
        if metrics:
            overall_retention = np.mean([m['retention_rate'] for m in metrics.values()])
            overall_loss = np.mean([m['uniqueness_loss'] for m in metrics.values()])
            
            metrics['overall'] = {
                'avg_retention_rate': overall_retention,
                'avg_uniqueness_loss': overall_loss,
                'information_preservation': overall_retention
            }
        
        return metrics
