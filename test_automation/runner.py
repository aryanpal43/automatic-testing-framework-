"""
Intelligent Automated Software Testing Framework - Test Automation Module
Test Execution Engine using Selenium WebDriver

This module executes generated test cases using Selenium WebDriver
and provides detailed execution reports.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, 
    WebDriverException, ElementClickInterceptedException
)
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Represents the result of a test execution"""
    test_id: str
    test_title: str
    status: str  # PASS, FAIL, SKIP, ERROR
    execution_time: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    steps_results: List[Dict[str, Any]] = None

@dataclass
class ExecutionReport:
    """Represents the complete execution report"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_execution_time: float
    start_time: datetime
    end_time: datetime
    test_results: List[TestResult]

class WebDriverManager:
    """Manages WebDriver initialization and configuration"""
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.driver = None
    
    def initialize_driver(self) -> webdriver.Chrome:
        """Initialize Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.implicitly_wait(10)
            logger.info("Chrome WebDriver initialized successfully")
            return self.driver
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def quit_driver(self):
        """Safely quit the WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")

class TestExecutor:
    """Executes test cases using Selenium WebDriver"""
    
    def __init__(self, base_url: str = "https://the-internet.herokuapp.com", headless: bool = False):
        self.base_url = base_url
        self.driver_manager = WebDriverManager(headless=headless)
        self.driver = None
        self.wait = None
        self.screenshot_dir = "data/screenshots"
        
        # Create screenshots directory if it doesn't exist
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    def setup_driver(self):
        """Setup WebDriver and WebDriverWait"""
        self.driver = self.driver_manager.initialize_driver()
        self.wait = WebDriverWait(self.driver, 10)
    
    def teardown_driver(self):
        """Cleanup WebDriver"""
        self.driver_manager.quit_driver()
    
    def take_screenshot(self, test_id: str, step_name: str = "") -> str:
        """Take a screenshot and save it"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_id}_{step_name}_{timestamp}.png"
        filepath = os.path.join(self.screenshot_dir, filename)
        
        try:
            self.driver.save_screenshot(filepath)
            logger.info(f"Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return ""
    
    def find_element_safe(self, selector: str, timeout: int = 10) -> Optional[Any]:
        """Safely find an element with timeout"""
        try:
            if selector.startswith('#'):
                return self.wait.until(EC.presence_of_element_located((By.ID, selector[1:])))
            elif selector.startswith('.'):
                return self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, selector[1:])))
            elif selector.startswith('//'):
                return self.wait.until(EC.presence_of_element_located((By.XPATH, selector)))
            else:
                return self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        except TimeoutException:
            logger.warning(f"Element not found: {selector}")
            return None
        except Exception as e:
            logger.error(f"Error finding element {selector}: {e}")
            return None
    
    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single test step"""
        step_result = {
            'step_number': step['step_number'],
            'action': step['action'],
            'status': 'PASS',
            'error_message': None,
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            action = step['action'].lower()
            selector = step.get('selector')
            input_data = step.get('input_data')
            
            if 'navigate' in action or 'open' in action:
                if selector == 'url':
                    # Navigate to base URL or specific page
                    if 'login' in action:
                        self.driver.get(f"{self.base_url}/login")
                    elif 'registration' in action or 'signup' in action:
                        self.driver.get(f"{self.base_url}/signup")
                    elif 'search' in action:
                        self.driver.get(f"{self.base_url}/search")
                    elif 'checkboxes' in action:
                        self.driver.get(f"{self.base_url}/checkboxes")
                    else:
                        self.driver.get(self.base_url)
                else:
                    self.driver.get(selector)
                
                logger.info(f"Navigated to: {self.driver.current_url}")
            
            elif 'enter' in action or 'input' in action:
                if selector and input_data:
                    element = self.find_element_safe(selector)
                    if element:
                        element.clear()
                        element.send_keys(input_data)
                        logger.info(f"Entered '{input_data}' in {selector}")
                    else:
                        raise Exception(f"Element not found: {selector}")
                else:
                    logger.warning("No selector or input data provided for input action")
            
            elif 'click' in action:
                if selector:
                    element = self.find_element_safe(selector)
                    if element:
                        # Scroll to element if needed
                        self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                        time.sleep(0.5)  # Brief pause for scroll
                        element.click()
                        logger.info(f"Clicked element: {selector}")
                    else:
                        raise Exception(f"Element not found: {selector}")
                else:
                    logger.warning("No selector provided for click action")
            
            elif 'verify' in action or 'check' in action:
                if selector:
                    element = self.find_element_safe(selector)
                    if not element:
                        raise Exception(f"Verification failed: Element not found {selector}")
                    logger.info(f"Verification passed: {selector} found")
                else:
                    logger.info("Verification step completed (no specific element to verify)")
            
            else:
                # Generic action - just log it
                logger.info(f"Executed action: {action}")
            
            step_result['execution_time'] = time.time() - start_time
            
        except Exception as e:
            step_result['status'] = 'FAIL'
            step_result['error_message'] = str(e)
            step_result['execution_time'] = time.time() - start_time
            logger.error(f"Step failed: {e}")
        
        return step_result
    
    def execute_test_case(self, test_case: Dict[str, Any]) -> TestResult:
        """Execute a single test case"""
        test_id = test_case['test_id']
        test_title = test_case['title']
        
        logger.info(f"Starting test execution: {test_id} - {test_title}")
        
        start_time = datetime.now()
        steps_results = []
        status = "PASS"
        error_message = None
        screenshot_path = None
        
        try:
            # Execute each step
            for step in test_case['steps']:
                step_result = self.execute_step(step)
                steps_results.append(step_result)
                
                # If any step fails, mark test as failed
                if step_result['status'] == 'FAIL':
                    status = "FAIL"
                    error_message = step_result['error_message']
                    screenshot_path = self.take_screenshot(test_id, f"step_{step['step_number']}")
                    break
                
                # Brief pause between steps
                time.sleep(0.5)
            
            # If all steps pass, take final screenshot
            if status == "PASS":
                screenshot_path = self.take_screenshot(test_id, "final")
        
        except Exception as e:
            status = "ERROR"
            error_message = str(e)
            screenshot_path = self.take_screenshot(test_id, "error")
            logger.error(f"Test execution error: {e}")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return TestResult(
            test_id=test_id,
            test_title=test_title,
            status=status,
            execution_time=execution_time,
            start_time=start_time,
            end_time=end_time,
            error_message=error_message,
            screenshot_path=screenshot_path,
            steps_results=steps_results
        )
    
    def run_tests(self, test_json_path: str = 'data/generated_tests.json') -> ExecutionReport:
        """Run all test cases from JSON file"""
        logger.info(f"Starting test execution from: {test_json_path}")
        
        # Load test cases
        try:
            with open(test_json_path, 'r') as f:
                test_cases = json.load(f)
        except FileNotFoundError:
            logger.error(f"Test file not found: {test_json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test file: {e}")
            raise
        
        # Setup WebDriver
        self.setup_driver()
        
        start_time = datetime.now()
        test_results = []
        
        try:
            # Execute each test case
            for test_case in test_cases:
                result = self.execute_test_case(test_case)
                test_results.append(result)
                
                # Brief pause between tests
                time.sleep(1)
        
        finally:
            # Always cleanup WebDriver
            self.teardown_driver()
        
        end_time = datetime.now()
        total_execution_time = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        passed_tests = sum(1 for r in test_results if r.status == "PASS")
        failed_tests = sum(1 for r in test_results if r.status == "FAIL")
        skipped_tests = sum(1 for r in test_results if r.status == "SKIP")
        
        report = ExecutionReport(
            total_tests=len(test_cases),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_execution_time,
            start_time=start_time,
            end_time=end_time,
            test_results=test_results
        )
        
        # Save detailed report
        self.save_report(report)
        
        logger.info(f"Test execution completed. Passed: {passed_tests}, Failed: {failed_tests}, Skipped: {skipped_tests}")
        return report
    
    def save_report(self, report: ExecutionReport, output_path: str = 'data/test_execution_report.json'):
        """Save execution report to JSON file"""
        report_dict = asdict(report)
        
        # Convert datetime objects to strings for JSON serialization
        report_dict['start_time'] = report.start_time.isoformat()
        report_dict['end_time'] = report.end_time.isoformat()
        
        for test_result in report_dict['test_results']:
            test_result['start_time'] = test_result['start_time'].isoformat()
            test_result['end_time'] = test_result['end_time'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Execution report saved to: {output_path}")

def run_tests(test_json_path='data/generated_tests.json'):
    """
    Legacy function for backward compatibility
    """
    executor = TestExecutor()
    report = executor.run_tests(test_json_path)
    
    # Print summary
    print(f"\n=== Test Execution Summary ===")
    print(f"Total Tests: {report.total_tests}")
    print(f"Passed: {report.passed_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Skipped: {report.skipped_tests}")
    print(f"Total Execution Time: {report.total_execution_time:.2f} seconds")
    
    # Print detailed results
    print(f"\n=== Detailed Results ===")
    for result in report.test_results:
        status_icon = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
        print(f"{status_icon} {result.test_id}: {result.test_title} ({result.status}) - {result.execution_time:.2f}s")
        if result.error_message:
            print(f"    Error: {result.error_message}")

if __name__ == '__main__':
    run_tests()
