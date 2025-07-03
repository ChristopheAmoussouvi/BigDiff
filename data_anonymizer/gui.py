"""
Streamlit GUI for the Data Anonymization Tool.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io
import json
from typing import Dict, List, Optional

from data_anonymizer.core.anonymizer import DataAnonymizer
from data_anonymizer.config.settings import AnonymizationConfig, PRIVACY_TEMPLATES

# Page configuration
st.set_page_config(
    page_title="Data Anonymizer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert > div {
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'anonymizer' not in st.session_state:
        st.session_state.anonymizer = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'anonymized_df' not in st.session_state:
        st.session_state.anonymized_df = None
    if 'anonymization_applied' not in st.session_state:
        st.session_state.anonymization_applied = False

def create_data_preview(df: pd.DataFrame, title: str) -> None:
    """Create a data preview section."""
    st.subheader(title)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # Show data types
    st.write("**Column Information:**")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null': df.count(),
        'Unique Values': df.nunique()
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Show sample data
    st.write("**Sample Data:**")
    st.dataframe(df.head(10), use_container_width=True)

def create_anonymization_controls() -> Dict:
    """Create the anonymization control panel."""
    st.sidebar.header("🔧 Anonymization Settings")
    
    # Privacy template selection
    template = st.sidebar.selectbox(
        "Choose Privacy Template:",
        options=["Custom"] + list(PRIVACY_TEMPLATES.keys()),
        help="Select a predefined privacy template or customize your own settings"
    )
    
    settings = {}
    
    if template != "Custom" and template in PRIVACY_TEMPLATES:
        template_settings = PRIVACY_TEMPLATES.get(template)
        if template_settings and isinstance(template_settings, dict):
            settings = dict(template_settings)  # Create a new dict instead of using copy()
            st.sidebar.info(f"Using {template.replace('_', ' ').title()} template")
    
    # Differential Privacy Settings
    st.sidebar.subheader("📊 Differential Privacy")
    
    epsilon = st.sidebar.slider(
        "Privacy Budget (ε)",
        min_value=0.01,
        max_value=5.0,
        value=settings.get('epsilon', 1.0),
        step=0.01,
        help="Lower values = more privacy, higher values = more utility"
    )
    
    # K-Anonymity Settings
    st.sidebar.subheader("🎭 K-Anonymity")
    
    k_value = st.sidebar.slider(
        "K-Value",
        min_value=2,
        max_value=20,
        value=settings.get('k', 2),
        help="Minimum group size for each combination of quasi-identifiers"
    )
    
    strategy = st.sidebar.selectbox(
        "Anonymization Strategy:",
        options=['suppression', 'generalization', 'synthetic'],
        index=['suppression', 'generalization', 'synthetic'].index(settings.get('strategy', 'generalization')),
        help="Choose how to handle records that violate k-anonymity"
    )
    
    return {
        'epsilon': epsilon,
        'k': k_value,
        'strategy': strategy
    }

def create_column_selector(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Create column selection interface."""
    st.subheader("📋 Column Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Columns (Differential Privacy)**")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numerical_cols:
            selected_numerical = st.multiselect(
                "Select numerical columns:",
                options=numerical_cols,
                default=[],
                help="These columns will have noise added for differential privacy"
            )
        else:
            st.info("No numerical columns detected")
            selected_numerical = []
    
    with col2:
        st.write("**Quasi-Identifiers (K-Anonymity)**")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            selected_categorical = st.multiselect(
                "Select quasi-identifier columns:",
                options=categorical_cols,
                default=[],
                help="These columns will be anonymized using k-anonymity"
            )
        else:
            st.info("No categorical columns detected")
            selected_categorical = []
    
    return {
        'numerical_columns': selected_numerical,
        'quasi_identifiers': selected_categorical
    }

def create_comparison_charts(original_df: pd.DataFrame, anonymized_df: pd.DataFrame, columns: List[str]):
    """Create comparison charts between original and anonymized data."""
    st.subheader("📊 Data Comparison")
    
    for col in columns[:3]:  # Limit to first 3 columns for performance
        if col in original_df.columns and col in anonymized_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Original {col}**")
                if pd.api.types.is_numeric_dtype(original_df[col]):
                    fig = px.histogram(original_df, x=col, title=f"Original {col} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = original_df[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Original {col} Top Values")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write(f"**Anonymized {col}**")
                if pd.api.types.is_numeric_dtype(anonymized_df[col]):
                    fig = px.histogram(anonymized_df, x=col, title=f"Anonymized {col} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = anonymized_df[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Anonymized {col} Top Values")
                    st.plotly_chart(fig, use_container_width=True)

def create_anonymization_report(anonymizer: DataAnonymizer):
    """Create an anonymization report display."""
    st.subheader("📋 Anonymization Report")
    
    report = anonymizer.get_anonymization_report()
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", report['metadata']['original_shape'][0])
    with col2:
        st.metric("Original Columns", report['metadata']['original_shape'][1])
    with col3:
        st.metric("Methods Applied", len(report['anonymization_log']))
    
    # Anonymization log
    if report['anonymization_log']:
        st.write("**Anonymization Methods Applied:**")
        for i, log_entry in enumerate(report['anonymization_log']):
            with st.expander(f"Method {i+1}: {log_entry['method'].replace('_', ' ').title()}"):
                st.json(log_entry)
    
    # Utility metrics
    if 'utility_metrics' in report:
        st.write("**Utility Metrics:**")
        utility_df = pd.DataFrame(report['utility_metrics']).T
        st.dataframe(utility_df, use_container_width=True)

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🔒 Data Anonymization Tool")
    st.markdown("**Protect sensitive data using Differential Privacy and K-Anonymity techniques**")
    
    # Sidebar controls
    anonymization_settings = create_anonymization_controls()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "⚙️ Configure", "🔄 Process", "📊 Results"])
    
    with tab1:
        st.header("Upload Your Dataset")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing the data you want to anonymize"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.session_state.original_df = df
                st.session_state.anonymizer = DataAnonymizer(random_seed=42)
                st.session_state.anonymizer.df = df.copy()
                st.session_state.anonymizer.original_df = df.copy()
                
                st.success(f"✅ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Data preview
                create_data_preview(df, "Dataset Preview")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with tab2:
        st.header("Configure Anonymization")
        
        if st.session_state.original_df is not None:
            # Column selection
            column_selection = create_column_selector(st.session_state.original_df)
            st.session_state.column_selection = column_selection
            
            # Show current settings
            st.subheader("🔧 Current Settings Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Differential Privacy:**
                - Epsilon: {anonymization_settings['epsilon']}
                - Columns: {', '.join(column_selection['numerical_columns']) if column_selection['numerical_columns'] else 'None selected'}
                """)
            
            with col2:
                st.info(f"""
                **K-Anonymity:**
                - K-Value: {anonymization_settings['k']}
                - Strategy: {anonymization_settings['strategy'].title()}
                - Columns: {', '.join(column_selection['quasi_identifiers']) if column_selection['quasi_identifiers'] else 'None selected'}
                """)
        else:
            st.warning("⚠️ Please upload a dataset first")
    
    with tab3:
        st.header("Process Data")
        
        if st.session_state.original_df is not None and hasattr(st.session_state, 'column_selection'):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("Click the button below to apply anonymization to your dataset:")
            
            with col2:
                if st.button("🚀 Apply Anonymization", type="primary"):
                    with st.spinner("Applying anonymization techniques..."):
                        try:
                            # Apply differential privacy
                            if st.session_state.column_selection['numerical_columns']:
                                st.session_state.anonymizer.apply_differential_privacy(
                                    st.session_state.column_selection['numerical_columns'],
                                    epsilon=anonymization_settings['epsilon']
                                )
                            
                            # Apply k-anonymity
                            if st.session_state.column_selection['quasi_identifiers']:
                                st.session_state.anonymizer.apply_k_anonymity(
                                    st.session_state.column_selection['quasi_identifiers'],
                                    k=anonymization_settings['k'],
                                    strategy=anonymization_settings['strategy']
                                )
                            
                            st.session_state.anonymized_df = st.session_state.anonymizer.df.copy()
                            st.session_state.anonymization_applied = True
                            
                            st.success("✅ Anonymization completed successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during anonymization: {str(e)}")
            
            # Show progress if anonymization is applied
            if st.session_state.anonymization_applied:
                st.subheader("📊 Processing Summary")
                
                # Create comparison preview
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data Sample:**")
                    st.dataframe(st.session_state.original_df.head(), use_container_width=True)
                
                with col2:
                    st.write("**Anonymized Data Sample:**")
                    st.dataframe(st.session_state.anonymized_df.head(), use_container_width=True)
        else:
            st.warning("⚠️ Please upload data and configure settings first")
    
    with tab4:
        st.header("Results & Analysis")
        
        if st.session_state.anonymization_applied:
            # Download section
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Download Anonymized Data:**")
            
            with col2:
                # Convert to CSV for download
                csv_buffer = io.StringIO()
                st.session_state.anonymized_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name="anonymized_data.csv",
                    mime="text/csv"
                )
            
            # Anonymization report
            create_anonymization_report(st.session_state.anonymizer)
            
            # Comparison charts
            all_columns = (
                st.session_state.column_selection.get('numerical_columns', []) +
                st.session_state.column_selection.get('quasi_identifiers', [])
            )
            
            if all_columns:
                create_comparison_charts(
                    st.session_state.original_df,
                    st.session_state.anonymized_df,
                    all_columns
                )
        else:
            st.info("🔄 Complete the anonymization process to view results")

if __name__ == "__main__":
    main()
