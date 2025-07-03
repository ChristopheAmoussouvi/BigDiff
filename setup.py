"""
Setup script for the Data Anonymizer tool.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="data-anonymizer-jace",
    version="1.0.0",
    author="JACE",
    author_email="contact@jace.com",
    description="A tool for data anonymization using Differential Privacy and K-Anonymity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jace-solutions/data-anonymizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "pandas",
        "numpy",
        "Faker",
    ],
    entry_points={
        'console_scripts': [
            'data-anonymizer=data_anonymizer.cli:main',
        ],
    },
)
