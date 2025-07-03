"""
Sample data generator for testing the Data Anonymization Tool.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

class SampleDataGenerator:
    """Generate realistic sample datasets for testing anonymization."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the sample data generator.
        
        Args:
            random_seed: Optional seed for reproducible data generation
        """
        self.fake = Faker()
        if random_seed:
            Faker.seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def generate_employee_dataset(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate a realistic employee dataset.
        
        Args:
            num_records: Number of employee records to generate
            
        Returns:
            DataFrame with employee data
        """
        data = []
        
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        positions = {
            'Engineering': ['Software Engineer', 'Senior Engineer', 'Tech Lead', 'Engineering Manager'],
            'Sales': ['Sales Rep', 'Account Manager', 'Sales Director', 'VP Sales'],
            'Marketing': ['Marketing Specialist', 'Content Manager', 'Marketing Director'],
            'HR': ['HR Specialist', 'Recruiter', 'HR Manager', 'HR Director'],
            'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'CFO'],
            'Operations': ['Operations Specialist', 'Project Manager', 'Operations Director']
        }
        
        for _ in range(num_records):
            # Basic demographics
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            age = np.random.randint(22, 65)
            gender = random.choice(['Male', 'Female', 'Other'])
            
            # Location data
            zipcode = self.fake.zipcode()
            city = self.fake.city()
            state = self.fake.state_abbr()
            
            # Employment data
            department = random.choice(departments)
            position = random.choice(positions[department])
            years_experience = max(0, age - 22 - np.random.randint(0, 5))
            
            # Salary based on position and experience
            base_salaries = {
                'Software Engineer': 80000, 'Senior Engineer': 120000, 'Tech Lead': 140000,
                'Engineering Manager': 160000, 'Sales Rep': 50000, 'Account Manager': 70000,
                'Sales Director': 130000, 'VP Sales': 180000, 'Marketing Specialist': 55000,
                'Content Manager': 65000, 'Marketing Director': 120000, 'HR Specialist': 50000,
                'Recruiter': 60000, 'HR Manager': 90000, 'HR Director': 130000,
                'Financial Analyst': 65000, 'Accountant': 55000, 'Finance Manager': 100000,
                'CFO': 200000, 'Operations Specialist': 55000, 'Project Manager': 85000,
                'Operations Director': 140000
            }
            
            base_salary = base_salaries.get(position, 60000)
            salary = int(base_salary + (years_experience * 2000) + np.random.normal(0, 10000))
            salary = max(30000, salary)  # Minimum salary
            
            # Education
            education_levels = ['High School', 'Associates', 'Bachelors', 'Masters', 'PhD']
            education_weights = [0.1, 0.15, 0.45, 0.25, 0.05]
            education = np.random.choice(education_levels, p=education_weights)
            
            # Contact information
            email = f"{first_name.lower()}.{last_name.lower()}@company.com"
            phone = self.fake.phone_number()
            
            # Employee ID
            employee_id = f"EMP{random.randint(10000, 99999)}"
            
            data.append({
                'employee_id': employee_id,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'age': age,
                'gender': gender,
                'email': email,
                'phone': phone,
                'zipcode': zipcode,
                'city': city,
                'state': state,
                'department': department,
                'position': position,
                'years_experience': years_experience,
                'education': education,
                'salary': salary
            })
        
        return pd.DataFrame(data)
    
    def generate_customer_dataset(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate a realistic customer dataset.
        
        Args:
            num_records: Number of customer records to generate
            
        Returns:
            DataFrame with customer data
        """
        data = []
        
        for _ in range(num_records):
            # Basic demographics
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            age = np.random.randint(18, 80)
            gender = random.choice(['Male', 'Female', 'Other'])
            
            # Location
            address = self.fake.street_address()
            city = self.fake.city()
            state = self.fake.state_abbr()
            zipcode = self.fake.zipcode()
            country = 'USA'
            
            # Contact
            email = self.fake.email()
            phone = self.fake.phone_number()
            
            # Financial data
            income = int(np.random.lognormal(10.5, 0.8))  # Log-normal distribution for income
            income = min(max(income, 20000), 500000)  # Cap between 20k and 500k
            
            credit_score = int(np.random.normal(650, 100))
            credit_score = min(max(credit_score, 300), 850)  # Credit score range
            
            # Customer behavior
            customer_since = self.fake.date_between(start_date='-10y', end_date='today')
            total_purchases = np.random.poisson(12)  # Average 12 purchases
            total_spent = int(np.random.exponential(500) * total_purchases)
            
            # Preferences
            preferred_category = random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'])
            marketing_opt_in = random.choice([True, False])
            
            # Customer ID
            customer_id = f"CUST{random.randint(100000, 999999)}"
            
            data.append({
                'customer_id': customer_id,
                'first_name': first_name,
                'last_name': last_name,
                'age': age,
                'gender': gender,
                'email': email,
                'phone': phone,
                'address': address,
                'city': city,
                'state': state,
                'zipcode': zipcode,
                'country': country,
                'income': income,
                'credit_score': credit_score,
                'customer_since': customer_since,
                'total_purchases': total_purchases,
                'total_spent': total_spent,
                'preferred_category': preferred_category,
                'marketing_opt_in': marketing_opt_in
            })
        
        return pd.DataFrame(data)
    
    def generate_medical_dataset(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate a synthetic medical dataset (HIPAA-like sensitive data).
        
        Args:
            num_records: Number of patient records to generate
            
        Returns:
            DataFrame with medical data
        """
        data = []
        
        conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Depression', 'Obesity', 
                     'Heart Disease', 'Arthritis', 'Cancer', 'Allergies', 'None']
        condition_weights = [0.15, 0.20, 0.12, 0.18, 0.25, 0.08, 0.15, 0.05, 0.30, 0.20]
        
        for _ in range(num_records):
            # Basic demographics
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            age = np.random.randint(18, 90)
            gender = random.choice(['Male', 'Female', 'Other'])
            
            # Location
            zipcode = self.fake.zipcode()
            state = self.fake.state_abbr()
            
            # Medical data
            patient_id = f"P{random.randint(100000, 999999)}"
            ssn = self.fake.ssn()
            
            # Health metrics
            height = np.random.normal(170, 15)  # cm
            weight = np.random.normal(70, 20)   # kg
            weight = max(weight, 40)  # Minimum weight
            bmi = weight / ((height/100) ** 2)
            
            blood_pressure_systolic = int(np.random.normal(120, 20))
            blood_pressure_diastolic = int(np.random.normal(80, 15))
            
            # Medical conditions (can have multiple)
            num_conditions = np.random.poisson(1.5)
            patient_conditions = np.random.choice(conditions, 
                                                size=min(num_conditions, 3), 
                                                replace=False,
                                                p=np.array(condition_weights)/sum(condition_weights))
            primary_condition = patient_conditions[0] if len(patient_conditions) > 0 else 'None'
            
            # Visit data
            last_visit = self.fake.date_between(start_date='-2y', end_date='today')
            visits_per_year = np.random.poisson(3)
            
            # Insurance
            insurance_types = ['Private', 'Medicare', 'Medicaid', 'Uninsured']
            insurance = random.choice(insurance_types)
            
            data.append({
                'patient_id': patient_id,
                'first_name': first_name,
                'last_name': last_name,
                'age': age,
                'gender': gender,
                'ssn': ssn,
                'zipcode': zipcode,
                'state': state,
                'height_cm': round(height, 1),
                'weight_kg': round(weight, 1),
                'bmi': round(bmi, 2),
                'bp_systolic': blood_pressure_systolic,
                'bp_diastolic': blood_pressure_diastolic,
                'primary_condition': primary_condition,
                'last_visit': last_visit,
                'visits_per_year': visits_per_year,
                'insurance_type': insurance
            })
        
        return pd.DataFrame(data)
    
    def generate_financial_dataset(self, num_records: int = 1000) -> pd.DataFrame:
        """
        Generate a synthetic financial dataset.
        
        Args:
            num_records: Number of financial records to generate
            
        Returns:
            DataFrame with financial data
        """
        data = []
        
        for _ in range(num_records):
            # Account holder info
            first_name = self.fake.first_name()
            last_name = self.fake.last_name()
            age = np.random.randint(18, 80)
            
            # Account details
            account_id = f"ACC{random.randint(1000000, 9999999)}"
            account_type = random.choice(['Checking', 'Savings', 'Credit', 'Investment'])
            
            # Financial data
            if account_type == 'Credit':
                balance = -abs(np.random.exponential(2000))  # Negative for credit
                credit_limit = abs(balance) + np.random.randint(1000, 10000)
            else:
                balance = np.random.exponential(5000)
                credit_limit = None
            
            # Transaction data
            monthly_transactions = np.random.poisson(25)
            avg_transaction = np.random.exponential(100)
            
            # Risk factors
            credit_score = int(np.random.normal(650, 100))
            credit_score = min(max(credit_score, 300), 850)
            
            # Location
            zipcode = self.fake.zipcode()
            state = self.fake.state_abbr()
            
            # Account history
            account_opened = self.fake.date_between(start_date='-10y', end_date='-1y')
            
            data.append({
                'account_id': account_id,
                'first_name': first_name,
                'last_name': last_name,
                'age': age,
                'zipcode': zipcode,
                'state': state,
                'account_type': account_type,
                'balance': round(balance, 2),
                'credit_limit': credit_limit,
                'credit_score': credit_score,
                'monthly_transactions': monthly_transactions,
                'avg_transaction_amount': round(avg_transaction, 2),
                'account_opened': account_opened
            })
        
        return pd.DataFrame(data)
    
    def save_sample_datasets(self, output_dir: str = "sample_data", num_records: int = 1000):
        """
        Generate and save all sample datasets.
        
        Args:
            output_dir: Directory to save the datasets
            num_records: Number of records for each dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating sample datasets with {num_records} records each...")
        
        # Generate datasets
        datasets = {
            'employees': self.generate_employee_dataset(num_records),
            'customers': self.generate_customer_dataset(num_records),
            'medical': self.generate_medical_dataset(num_records),
            'financial': self.generate_financial_dataset(num_records)
        }
        
        # Save datasets
        for name, df in datasets.items():
            file_path = output_path / f"{name}_sample_data.csv"
            df.to_csv(file_path, index=False)
            print(f"✅ Saved {name} dataset: {file_path} ({len(df)} records, {len(df.columns)} columns)")
        
        print(f"\n📁 All datasets saved to: {output_path.absolute()}")
        return datasets

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate sample datasets for anonymization testing")
    parser.add_argument("--records", "-n", type=int, default=1000, 
                       help="Number of records to generate (default: 1000)")
    parser.add_argument("--output", "-o", type=str, default="sample_data",
                       help="Output directory (default: sample_data)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset", "-d", type=str, choices=['employees', 'customers', 'medical', 'financial', 'all'],
                       default='all', help="Which dataset to generate (default: all)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = SampleDataGenerator(random_seed=args.seed)
    
    if args.dataset == 'all':
        # Generate all datasets
        generator.save_sample_datasets(args.output, args.records)
    else:
        # Generate specific dataset
        output_path = Path(args.output)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating {args.dataset} dataset with {args.records} records...")
        
        df = None
        if args.dataset == 'employees':
            df = generator.generate_employee_dataset(args.records)
        elif args.dataset == 'customers':
            df = generator.generate_customer_dataset(args.records)
        elif args.dataset == 'medical':
            df = generator.generate_medical_dataset(args.records)
        elif args.dataset == 'financial':
            df = generator.generate_financial_dataset(args.records)
        
        if df is not None:
            file_path = output_path / f"{args.dataset}_sample_data.csv"
            df.to_csv(file_path, index=False)
            print(f"✅ Saved dataset: {file_path} ({len(df)} records, {len(df.columns)} columns)")

if __name__ == "__main__":
    main()
