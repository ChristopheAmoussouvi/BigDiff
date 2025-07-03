# Data Anonymizer Tool

A powerful and flexible tool for anonymizing datasets using Differential Privacy and K-Anonymity techniques with a modern Streamlit GUI.

## Features

- **🔒 Differential Privacy**: Apply mathematically rigorous Laplace or Gaussian noise to numerical columns
- **🎭 K-Anonymity**: Anonymize categorical data using suppression, generalization, or synthetic data generation
- **🖥️ Modern GUI**: Interactive Streamlit web interface for easy data upload and configuration
- **⚙️ Flexible Configuration**: Use predefined privacy templates or customize settings
- **📊 Visual Analytics**: Compare original vs anonymized data with interactive charts
- **📋 Comprehensive Reporting**: Detailed anonymization reports and utility metrics
- **🔧 Multiple Interfaces**: GUI, CLI, and Python library interfaces
- **📈 Privacy Templates**: Pre-configured settings for different privacy levels

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd data-anonymizer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

## Usage

### 🖥️ Streamlit GUI (Recommended)

Launch the interactive web interface:

```bash
streamlit run data_anonymizer/gui.py
```

The GUI provides:
- **📁 Easy file upload** - Drag and drop CSV files
- **⚙️ Interactive controls** - Adjust epsilon and k-values with sliders
- **🎯 Column selection** - Choose which columns to anonymize
- **📊 Live previews** - See your data before and after anonymization
- **📥 One-click download** - Export anonymized data instantly

### 💻 Command-Line Interface

```bash
data-anonymizer input.csv output.csv --numerical_cols age salary --quasi_identifiers zipcode --k 5 --epsilon 1.0
```

**Options:**
- `--numerical_cols`: Columns for differential privacy
- `--quasi_identifiers`: Columns for k-anonymity
- `--epsilon`: Privacy budget (lower = more private)
- `--k`: Minimum group size for k-anonymity
- `--strategy`: K-anonymity strategy (suppression, generalization, synthetic)
- `--template`: Use privacy template (high_privacy, medium_privacy, low_privacy)

### 🐍 Python Library

```python
from data_anonymizer import DataAnonymizer

# Initialize the anonymizer
anonymizer = DataAnonymizer(random_seed=42)

# Load data
anonymizer.load_data("input.csv")

# Apply differential privacy to numerical columns
anonymizer.apply_differential_privacy(
    numerical_columns=['age', 'salary'], 
    epsilon=1.0
)

# Apply k-anonymity to categorical columns
anonymizer.apply_k_anonymity(
    quasi_identifiers=['zipcode', 'gender'], 
    k=5, 
    strategy='generalization'
)

# Save anonymized data with report
anonymizer.save_anonymized_data("output.csv", include_report=True)

# Get detailed analytics
report = anonymizer.get_anonymization_report()
print(report)
```

## Privacy Templates

Choose from predefined privacy levels:

- **🔒 High Privacy**: ε=0.1, k=5, synthetic strategy
- **⚖️ Medium Privacy**: ε=1.0, k=3, generalization strategy  
- **🔓 Low Privacy**: ε=2.0, k=2, suppression strategy
- **🎓 Research Compliant**: ε=0.5, k=3, balanced approach
- **⚡ Minimal**: ε=5.0, k=2, minimal anonymization

## Sample Data Generation

The tool includes a powerful sample data generator to create realistic datasets for testing.

**Generate all sample datasets:**
```bash
python -m data_anonymizer.sample_generator
```

**Generate a specific dataset:**
```bash
python -m data_anonymizer.sample_generator --dataset medical --records 500
```

**Available datasets:**
- `employees`: Corporate employee data
- `customers`: Retail customer data
- `medical`: Sensitive patient data (HIPAA-like)
- `financial`: Financial account data

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest

# Run with coverage
pytest --cov=data_anonymizer
```

## Project Structure

```
data_anonymizer/
├── core/                 # Core anonymization logic
│   ├── anonymizer.py    # Main anonymizer class
│   ├── privacy.py       # Differential privacy implementation
│   └── kanonymity.py    # K-anonymity implementation
├── config/              # Configuration management
│   └── settings.py      # Settings and templates
├── gui.py              # Streamlit web interface
└── cli.py              # Command-line interface

tests/                   # Test suite
├── test_anonymizer.py
├── test_privacy.py
├── test_kanonymity.py
└── test_config.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{data_anonymizer,
  title={Data Anonymizer Tool},
  author={JACE Studio},
  year={2025},
  url={https://github.com/jace-solutions/data-anonymizer}
}
