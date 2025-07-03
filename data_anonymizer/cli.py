"""
Command-Line Interface for the Data Anonymization Tool.
"""

import argparse
import logging
from pathlib import Path
from data_anonymizer.core.anonymizer import DataAnonymizer
from data_anonymizer.config.settings import AnonymizationConfig, create_config_from_template

def main():
    """Main function for the CLI."""
    parser = argparse.ArgumentParser(description="Data Anonymization Tool")
    
    # Input/Output arguments
    parser.add_argument("input_file", type=Path, help="Path to the input data file")
    parser.add_argument("output_file", type=Path, help="Path to save the anonymized data")
    
    # Anonymization parameters
    parser.add_argument("--numerical_cols", nargs='+', help="Numerical columns for differential privacy")
    parser.add_argument("--quasi_identifiers", nargs='+', help="Quasi-identifiers for k-anonymity")
    
    # Configuration
    parser.add_argument("--config", type=Path, help="Path to a custom config file")
    parser.add_argument("--template", choices=['high_privacy', 'medium_privacy', 'low_privacy'],
                        help="Use a predefined privacy template")
    
    # Privacy settings overrides
    parser.add_argument("--epsilon", type=float, help="Override epsilon value")
    parser.add_argument("--k", type=int, help="Override k value for k-anonymity")
    parser.add_argument("--strategy", choices=['suppression', 'generalization', 'synthetic'],
                        help="Override k-anonymity strategy")
    
    # Other options
    parser.add_argument("--no-report", action="store_true", help="Do not generate an anonymization report")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    if args.config:
        config = AnonymizationConfig(config_path=args.config)
    elif args.template:
        config = create_config_from_template(args.template)
    else:
        config = AnonymizationConfig()
    
    # Override config with CLI arguments
    updates = {}
    if args.epsilon:
        updates.setdefault('privacy_settings', {})['epsilon'] = args.epsilon
    if args.k:
        updates.setdefault('privacy_settings', {})['k'] = args.k
    if args.strategy:
        updates.setdefault('privacy_settings', {})['strategy'] = args.strategy
    
    if updates:
        config.update_config(updates)
    
    # Initialize anonymizer
    anonymizer = DataAnonymizer(config=config, random_seed=args.seed)
    
    try:
        # Load data
        logging.info(f"Loading data from {args.input_file}")
        anonymizer.load_data(args.input_file)
        
        # Apply anonymization
        if args.numerical_cols:
            logging.info("Applying Differential Privacy...")
            anonymizer.apply_differential_privacy(args.numerical_cols)
        
        if args.quasi_identifiers:
            logging.info("Applying K-Anonymity...")
            anonymizer.apply_k_anonymity(args.quasi_identifiers)
        
        # Save data
        logging.info(f"Saving anonymized data to {args.output_file}")
        anonymizer.save_anonymized_data(args.output_file, include_report=not args.no_report)
        
        # Print report summary
        anonymizer.get_anonymization_report(detailed=False)
        
        logging.info("Anonymization complete.")
        
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
