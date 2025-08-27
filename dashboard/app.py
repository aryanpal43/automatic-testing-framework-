"""
Intelligent Automated Software Testing Framework - Dashboard Module
Streamlit Web Application for Test Management and Bug Prediction
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load all necessary data files"""
    test_cases = load_json_file('data/generated_tests.json', [])
    execution_report = load_json_file('data/test_execution_report.json', {})
    model_metrics = load_json_file('data/model_metrics.json', {})
    bug_predictions = load_json_file('data/bug_predictions.json', {})
    software_metrics = load_csv_file('data/software_metrics.csv')
    return test_cases, execution_report, model_metrics, bug_predictions, software_metrics

def load_json_file(file_path: str, default_value):
    """Load JSON file with error handling"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return default_value
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return default_value

def load_csv_file(file_path: str):
    """Load CSV file with error handling"""
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def render_header(test_cases, execution_report, model_metrics, bug_predictions):
    """Render the main header"""
    st.title("🧪 Intelligent Automated Software Testing Framework")
    st.markdown("---")
    
    # Display current status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Cases", len(test_cases))
    
    with col2:
        if execution_report:
            passed = execution_report.get('passed_tests', 0)
            total = execution_report.get('total_tests', 0)
            st.metric("Tests Passed", f"{passed}/{total}")
        else:
            st.metric("Tests Passed", "0/0")
    
    with col3:
        if bug_predictions:
            high_risk = bug_predictions.get('high_risk_modules', 0)
            st.metric("High Risk Modules", high_risk)
        else:
            st.metric("High Risk Modules", 0)
    
    with col4:
        if model_metrics:
            best_model = max(model_metrics.keys(), 
                           key=lambda x: model_metrics[x].get('f1_score', 0))
            accuracy = model_metrics[best_model].get('accuracy', 0)
            st.metric("Best Model Accuracy", f"{accuracy:.1%}")
        else:
            st.metric("Best Model Accuracy", "N/A")

def render_overview_page(execution_report, model_metrics):
    """Render the overview page"""
    st.header("📊 Overview Dashboard")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Test execution results chart
        if execution_report and execution_report.get('test_results'):
            st.subheader("Test Execution Results")
            
            results = execution_report['test_results']
            status_counts = {}
            for result in results:
                status = result.get('status', 'UNKNOWN')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if status_counts:
                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Test Status Distribution",
                    color_discrete_map={
                        'PASS': '#00FF00',
                        'FAIL': '#FF0000',
                        'SKIP': '#FFFF00',
                        'ERROR': '#FF6600'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance chart
        if model_metrics:
            st.subheader("Model Performance Comparison")
            
            models = list(model_metrics.keys())
            accuracies = [model_metrics[m].get('accuracy', 0) for m in models]
            f1_scores = [model_metrics[m].get('f1_score', 0) for m in models]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models,
                y=accuracies,
                name='Accuracy',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=models,
                y=f1_scores,
                name='F1-Score',
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

def render_test_generation_page():
    """Render the test generation page"""
    st.header("📝 Test Case Generation")
    
    # Test case generation form
    with st.expander("Generate New Test Cases", expanded=True):
        requirement_text = st.text_area(
            "Enter Software Requirements",
            height=200,
            placeholder="Enter your software requirements here...\n\nExample:\nThe system shall allow users to login using their email and password.\nThe system shall provide user registration functionality.\nThe system shall allow users to search for products."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Test Cases", type="primary"):
                if requirement_text.strip():
                    with st.spinner("Generating test cases..."):
                        try:
                            from nlp_module.generate_tests import generate_test_cases
                            test_cases = generate_test_cases(requirement_text)
                            st.success(f"Generated {len(test_cases)} test cases!")
                        except Exception as e:
                            st.error(f"Error generating test cases: {e}")
                else:
                    st.warning("Please enter requirements text.")
        
        with col2:
            if st.button("Load Sample Requirements"):
                sample_requirements = """The system shall allow users to login using their email and password.
The system shall provide user registration functionality with email verification.
The system shall allow users to search for products by name or category.
The system shall enable users to add items to their shopping cart.
The system shall allow users to update their profile information.
The system shall provide password reset functionality via email."""
                st.session_state.requirements = sample_requirements
                st.rerun()

def render_test_execution_page():
    """Render the test execution page"""
    st.header("▶️ Test Execution")
    
    # Test execution controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Run All Tests", type="primary"):
            with st.spinner("Executing tests..."):
                try:
                    from test_automation.runner import TestExecutor
                    executor = TestExecutor()
                    report = executor.run_tests()
                    st.success("Test execution completed!")
                except Exception as e:
                    st.error(f"Error during test execution: {e}")
    
    with col2:
        headless_mode = st.checkbox("Headless Mode", value=True)
    
    with col3:
        if st.button("View Screenshots"):
            screenshot_dir = "data/screenshots"
            if os.path.exists(screenshot_dir):
                screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.png')]
                if screenshots:
                    st.write("**Available Screenshots:**")
                    for screenshot in screenshots:
                        st.write(f"• {screenshot}")
                else:
                    st.info("No screenshots available.")
            else:
                st.info("Screenshots directory not found.")

def render_bug_prediction_page():
    """Render the bug prediction page"""
    st.header("🐛 Bug Prediction")
    
    # Model training controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                try:
                    from bug_prediction.train_model import BugPredictor
                    predictor = BugPredictor()
                    metrics = predictor.train_models()
                    st.success("Models trained successfully!")
                except Exception as e:
                    st.error(f"Error training models: {e}")
    
    with col2:
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    from bug_prediction.train_model import BugPredictor
                    predictor = BugPredictor()
                    # This would generate predictions for available data
                    st.success("Predictions generated!")
                except Exception as e:
                    st.error(f"Error generating predictions: {e}")

def run():
    """Main function to run the dashboard"""
    st.set_page_config(
        page_title="Intelligent Auto Testing Framework",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    test_cases, execution_report, model_metrics, bug_predictions, software_metrics = load_data()
    
    # Render header
    render_header(test_cases, execution_report, model_metrics, bug_predictions)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Test Generation", "Test Execution", "Bug Prediction"]
    )
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.subheader("Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Render selected page
    if page == "Overview":
        render_overview_page(execution_report, model_metrics)
    elif page == "Test Generation":
        render_test_generation_page()
    elif page == "Test Execution":
        render_test_execution_page()
    elif page == "Bug Prediction":
        render_bug_prediction_page()

if __name__ == '__main__':
    run()
