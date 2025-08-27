"""
Intelligent Automated Software Testing Framework - Main Pipeline
Integrated pipeline that combines all modules for end-to-end testing

This script provides a complete workflow:
1. Generate test cases from requirements
2. Execute tests using Selenium
3. Train bug prediction models
4. Generate predictions
5. Display results in dashboard
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data',
        'data/screenshots',
        'data/models',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def run_nlp_pipeline(requirements_text: str, output_path: str = 'data/generated_tests.json'):
    """Run NLP test generation pipeline"""
    logger.info("Starting NLP test generation pipeline...")
    
    try:
        from nlp_module.generate_tests import generate_test_cases
        
        test_cases = generate_test_cases(requirements_text, output_path)
        logger.info(f"NLP pipeline completed. Generated {len(test_cases)} test cases.")
        return test_cases
    
    except Exception as e:
        logger.error(f"Error in NLP pipeline: {e}")
        raise

def run_test_execution_pipeline(test_file: str = 'data/generated_tests.json', headless: bool = True):
    """Run test execution pipeline"""
    logger.info("Starting test execution pipeline...")
    
    try:
        from test_automation.runner import TestExecutor
        
        executor = TestExecutor(headless=headless)
        report = executor.run_tests(test_file)
        
        logger.info(f"Test execution completed. Passed: {report.passed_tests}, Failed: {report.failed_tests}")
        return report
    
    except Exception as e:
        logger.error(f"Error in test execution pipeline: {e}")
        raise

def run_bug_prediction_pipeline(data_path: str = 'data/software_metrics.csv'):
    """Run bug prediction pipeline"""
    logger.info("Starting bug prediction pipeline...")
    
    try:
        from bug_prediction.train_model import BugPredictor
        
        predictor = BugPredictor()
        metrics = predictor.train_models(data_path)
        
        # Generate predictions for sample modules
        if os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
            sample_modules = df.head(10).to_dict('records')
            predictions = predictor.predict_multiple_modules(sample_modules)
            predictor.generate_prediction_report(predictions)
            
            logger.info(f"Bug prediction completed. Generated predictions for {len(predictions)} modules.")
            return predictor, predictions
        
        logger.info("Bug prediction pipeline completed.")
        return predictor, []
    
    except Exception as e:
        logger.error(f"Error in bug prediction pipeline: {e}")
        raise

def run_dashboard():
    """Start the Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard...")
    
    try:
        import subprocess
        import webbrowser
        import time
        
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
            "--server.port", "8501", "--server.headless", "true"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        logger.info("Dashboard started at http://localhost:8501")
        return process
    
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        raise

def run_complete_pipeline(requirements_text: str, headless: bool = True):
    """Run the complete pipeline from requirements to dashboard"""
    logger.info("Starting complete testing pipeline...")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Setup
        setup_directories()
        
        # Step 2: Generate test cases
        test_cases = run_nlp_pipeline(requirements_text)
        
        # Step 3: Execute tests
        execution_report = run_test_execution_pipeline(headless=headless)
        
        # Step 4: Train models and generate predictions
        predictor, predictions = run_bug_prediction_pipeline()
        
        # Step 5: Generate summary report
        generate_summary_report(test_cases, execution_report, predictions)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Complete pipeline finished in {duration:.2f} seconds")
        
        return {
            'test_cases': test_cases,
            'execution_report': execution_report,
            'predictions': predictions,
            'duration': duration
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def generate_summary_report(test_cases, execution_report, predictions):
    """Generate a comprehensive summary report"""
    logger.info("Generating summary report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_test_cases': len(test_cases),
            'tests_executed': execution_report.total_tests if execution_report else 0,
            'tests_passed': execution_report.passed_tests if execution_report else 0,
            'tests_failed': execution_report.failed_tests if execution_report else 0,
            'modules_analyzed': len(predictions),
            'high_risk_modules': len([p for p in predictions if p.bug_probability > 0.7]) if predictions else 0
        },
        'test_cases': [
            {
                'id': tc.test_id,
                'title': tc.title,
                'priority': tc.priority
            } for tc in test_cases
        ] if test_cases else [],
        'execution_results': {
            'total_tests': execution_report.total_tests if execution_report else 0,
            'passed_tests': execution_report.passed_tests if execution_report else 0,
            'failed_tests': execution_report.failed_tests if execution_report else 0,
            'execution_time': execution_report.total_execution_time if execution_report else 0
        } if execution_report else {},
        'bug_predictions': {
            'total_modules': len(predictions),
            'high_risk': len([p for p in predictions if p.bug_probability > 0.7]),
            'medium_risk': len([p for p in predictions if 0.3 <= p.bug_probability <= 0.7]),
            'low_risk': len([p for p in predictions if p.bug_probability < 0.3])
        } if predictions else {}
    }
    
    # Save report
    import json
    with open('data/pipeline_summary.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("Summary report saved to data/pipeline_summary.json")
    return report

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Intelligent Automated Software Testing Framework")
    parser.add_argument('--requirements', '-r', type=str, help='Path to requirements file')
    parser.add_argument('--requirements-text', '-t', type=str, help='Requirements text directly')
    parser.add_argument('--headless', action='store_true', default=True, help='Run tests in headless mode')
    parser.add_argument('--dashboard', action='store_true', help='Start dashboard after pipeline')
    parser.add_argument('--nlp-only', action='store_true', help='Run only NLP test generation')
    parser.add_argument('--test-only', action='store_true', help='Run only test execution')
    parser.add_argument('--prediction-only', action='store_true', help='Run only bug prediction')
    
    args = parser.parse_args()
    
    # Get requirements text
    requirements_text = ""
    if args.requirements:
        with open(args.requirements, 'r') as f:
            requirements_text = f.read()
    elif args.requirements_text:
        requirements_text = args.requirements_text
    else:
        # Default sample requirements
        requirements_text = """The system shall allow users to login using their email and password.
The system shall provide user registration functionality with email verification.
The system shall allow users to search for products by name or category.
The system shall enable users to add items to their shopping cart.
The system shall allow users to update their profile information.
The system shall provide password reset functionality via email."""
    
    try:
        if args.nlp_only:
            logger.info("Running NLP pipeline only...")
            run_nlp_pipeline(requirements_text)
        
        elif args.test_only:
            logger.info("Running test execution only...")
            run_test_execution_pipeline(headless=args.headless)
        
        elif args.prediction_only:
            logger.info("Running bug prediction only...")
            run_bug_prediction_pipeline()
        
        else:
            # Run complete pipeline
            results = run_complete_pipeline(requirements_text, args.headless)
            
            # Print summary
            print("\n" + "="*60)
            print("PIPELINE SUMMARY")
            print("="*60)
            print(f"Test Cases Generated: {len(results['test_cases'])}")
            print(f"Tests Executed: {results['execution_report'].total_tests}")
            print(f"Tests Passed: {results['execution_report'].passed_tests}")
            print(f"Tests Failed: {results['execution_report'].failed_tests}")
            print(f"Modules Analyzed: {len(results['predictions'])}")
            print(f"High Risk Modules: {len([p for p in results['predictions'] if p.bug_probability > 0.7])}")
            print(f"Total Duration: {results['duration']:.2f} seconds")
            print("="*60)
        
        # Start dashboard if requested
        if args.dashboard:
            run_dashboard()
            input("Press Enter to stop the dashboard...")
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 