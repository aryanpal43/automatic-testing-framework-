#!/usr/bin/env python3
"""
Intelligent Automated Software Testing Framework - Demo Script
Demonstrates the complete functionality of the framework
"""

import os
import sys
import time
from datetime import datetime

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    🧪 INTELLIGENT AUTO TESTING FRAMEWORK 🧪                    ║
    ║                                                                              ║
    ║  Final Year Major Project - Aryan Pal (7th Semester)                        ║
    ║  Supervisor: Ms. Richa Gupta                                                ║
    ║                                                                              ║
    ║  AI-Powered Software Testing with Bug Prediction                            ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def demo_nlp_module():
    """Demonstrate NLP test generation"""
    print("\n" + "="*60)
    print("🤖 DEMO: NLP Test Generation")
    print("="*60)
    
    sample_requirements = """
    The system shall allow users to login using their email and password.
    The system shall provide user registration functionality with email verification.
    The system shall allow users to search for products by name or category.
    The system shall enable users to add items to their shopping cart.
    The system shall allow users to update their profile information.
    The system shall provide password reset functionality via email.
    """
    
    print("📝 Sample Requirements:")
    print(sample_requirements.strip())
    
    try:
        from nlp_module.generate_tests import generate_test_cases
        
        print("\n🔄 Generating test cases...")
        test_cases = generate_test_cases(sample_requirements)
        
        print(f"✅ Generated {len(test_cases)} test cases successfully!")
        
        # Show first test case
        if test_cases:
            first_tc = test_cases[0]
            print(f"\n📋 Example Test Case:")
            print(f"   ID: {first_tc.test_id}")
            print(f"   Title: {first_tc.title}")
            print(f"   Steps: {len(first_tc.steps)}")
            for i, step in enumerate(first_tc.steps[:3], 1):
                print(f"   {i}. {step.action}")
            if len(first_tc.steps) > 3:
                print(f"   ... and {len(first_tc.steps) - 3} more steps")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in NLP module: {e}")
        return False

def demo_test_execution():
    """Demonstrate test execution"""
    print("\n" + "="*60)
    print("🔄 DEMO: Test Execution")
    print("="*60)
    
    try:
        from test_automation.runner import TestExecutor
        
        print("🚀 Starting test execution...")
        executor = TestExecutor(headless=True)
        report = executor.run_tests('data/sample_test_cases.json')
        
        print(f"✅ Test execution completed!")
        print(f"   Total Tests: {report.total_tests}")
        print(f"   Passed: {report.passed_tests}")
        print(f"   Failed: {report.failed_tests}")
        print(f"   Execution Time: {report.total_execution_time:.2f} seconds")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in test execution: {e}")
        return False

def demo_bug_prediction():
    """Demonstrate bug prediction"""
    print("\n" + "="*60)
    print("🐛 DEMO: Bug Prediction")
    print("="*60)
    
    try:
        from bug_prediction.train_model import BugPredictor
        
        print("🤖 Training ML models...")
        predictor = BugPredictor()
        metrics = predictor.train_models()
        
        print(f"✅ Models trained successfully!")
        
        # Show best model
        best_model = max(metrics.keys(), key=lambda x: metrics[x].f1_score)
        best_metrics = metrics[best_model]
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Accuracy: {best_metrics.accuracy:.1%}")
        print(f"   F1-Score: {best_metrics.f1_score:.1%}")
        print(f"   ROC-AUC: {best_metrics.roc_auc:.1%}")
        
        # Demo prediction
        print(f"\n🔮 Making sample prediction...")
        sample_module = {
            'module_name': 'demo_module',
            'lines_of_code': 800,
            'cyclomatic_complexity': 20,
            'number_of_functions': 30,
            'number_of_classes': 8,
            'depth_of_inheritance': 4,
            'coupling_between_objects': 15,
            'lack_of_cohesion': 6,
            'number_of_parameters': 8,
            'number_of_variables': 25,
            'number_of_comments': 80,
            'code_duplication': 0.15,
            'test_coverage': 0.6,
            'code_churn': 35,
            'developer_experience': 4,
            'module_age_days': 150
        }
        
        prediction = predictor.predict_bugs(sample_module)
        print(f"   Module: {prediction.module_name}")
        print(f"   Bug Probability: {prediction.bug_probability:.1%}")
        print(f"   Prediction: {'🐛 Bug' if prediction.prediction == 1 else '✅ No Bug'}")
        print(f"   Confidence: {prediction.confidence:.1%}")
        
        return True
    
    except Exception as e:
        print(f"❌ Error in bug prediction: {e}")
        return False

def demo_dashboard():
    """Demonstrate dashboard"""
    print("\n" + "="*60)
    print("📊 DEMO: Dashboard")
    print("="*60)
    
    print("🌐 Starting Streamlit dashboard...")
    print("   Dashboard will open in your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard")
    
    try:
        import subprocess
        import webbrowser
        import time
        
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
            "--server.port", "8501", "--server.headless", "true"
        ])
        
        # Wait for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        print("✅ Dashboard started successfully!")
        print("   Navigate through the different pages to explore:")
        print("   - Overview: See test results and model performance")
        print("   - Test Generation: Generate new test cases")
        print("   - Test Execution: Run and monitor tests")
        print("   - Bug Prediction: View predictions and model metrics")
        
        return process
    
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def main():
    """Main demo function"""
    print_banner()
    
    print("🎯 This demo will showcase all components of the framework:")
    print("   1. 🤖 NLP Test Generation")
    print("   2. 🔄 Test Execution")
    print("   3. 🐛 Bug Prediction")
    print("   4. 📊 Dashboard")
    
    input("\nPress Enter to start the demo...")
    
    # Run demos
    results = []
    
    # Demo 1: NLP
    results.append(demo_nlp_module())
    
    # Demo 2: Test Execution
    results.append(demo_test_execution())
    
    # Demo 3: Bug Prediction
    results.append(demo_bug_prediction())
    
    # Demo 4: Dashboard
    dashboard_process = demo_dashboard()
    
    # Summary
    print("\n" + "="*60)
    print("📋 DEMO SUMMARY")
    print("="*60)
    
    successful_demos = sum(results)
    total_demos = len(results)
    
    print(f"✅ Successful demos: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("🎉 All demos completed successfully!")
        print("\n🚀 The framework is ready for use!")
        print("\n📚 Next steps:")
        print("   1. Explore the dashboard at http://localhost:8501")
        print("   2. Try generating test cases with your own requirements")
        print("   3. Run the complete pipeline: python main.py --dashboard")
        print("   4. Check the documentation in README.md")
    else:
        print("⚠️  Some demos failed. Check the error messages above.")
    
    if dashboard_process:
        print(f"\n🌐 Dashboard is running. Press Ctrl+C to stop...")
        try:
            dashboard_process.wait()
        except KeyboardInterrupt:
            print("\n👋 Demo completed. Thank you!")

if __name__ == '__main__':
    main() 