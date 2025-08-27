"""
Intelligent Automated Software Testing Framework - NLP Module
Test Case Generation using Natural Language Processing

This module processes software requirement documents and generates
functional test cases using NLP techniques.
"""

import json
import re
import spacy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestStep:
    """Represents a single test step with action and expected result"""
    step_number: int
    action: str
    expected_result: str
    selector: Optional[str] = None
    input_data: Optional[str] = None

@dataclass
class TestCase:
    """Represents a complete test case"""
    test_id: str
    title: str
    description: str
    priority: str
    steps: List[TestStep]
    preconditions: List[str]
    test_data: Dict[str, Any]

class RequirementParser:
    """Parses software requirements using NLP techniques"""
    
    def __init__(self):
        try:
            # Load English language model for spaCy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from requirement text"""
        doc = self.nlp(text)
        entities = {
            'actions': [],
            'objects': [],
            'conditions': [],
            'validations': []
        }
        
        for token in doc:
            # Extract action verbs
            if token.pos_ == "VERB":
                entities['actions'].append(token.lemma_)
            
            # Extract objects (nouns)
            elif token.pos_ in ["NOUN", "PROPN"]:
                entities['objects'].append(token.text)
        
        # Extract conditions and validations using patterns
        conditions = re.findall(r'if\s+(.+?)(?:\s+then|\s*,)', text, re.IGNORECASE)
        entities['conditions'].extend(conditions)
        
        validations = re.findall(r'(?:should|must|shall)\s+(.+?)(?:\s+and|\s+or|\.)', text, re.IGNORECASE)
        entities['validations'].extend(validations)
        
        return entities

class TestCaseGenerator:
    """Generates test cases from parsed requirements"""
    
    def __init__(self):
        self.parser = RequirementParser()
        self.nlp = self.parser.nlp  # Access the spaCy model from parser
        self.test_patterns = {
            'login': {
                'title': 'User Login Test',
                'steps': [
                    {'action': 'Navigate to login page', 'selector': 'url', 'input': None},
                    {'action': 'Enter username', 'selector': '#username', 'input': 'test_user'},
                    {'action': 'Enter password', 'selector': '#password', 'input': 'test_password'},
                    {'action': 'Click login button', 'selector': '#login-button', 'input': None},
                    {'action': 'Verify successful login', 'selector': '.dashboard', 'input': None}
                ]
            },
            'register': {
                'title': 'User Registration Test',
                'steps': [
                    {'action': 'Navigate to registration page', 'selector': 'url', 'input': None},
                    {'action': 'Enter email', 'selector': '#email', 'input': 'test@example.com'},
                    {'action': 'Enter password', 'selector': '#password', 'input': 'TestPass123!'},
                    {'action': 'Enter confirm password', 'selector': '#confirm-password', 'input': 'TestPass123!'},
                    {'action': 'Click register button', 'selector': '#register-button', 'input': None},
                    {'action': 'Verify successful registration', 'selector': '.success-message', 'input': None}
                ]
            },
            'search': {
                'title': 'Search Functionality Test',
                'steps': [
                    {'action': 'Navigate to search page', 'selector': 'url', 'input': None},
                    {'action': 'Enter search term', 'selector': '#search-input', 'input': 'test search'},
                    {'action': 'Click search button', 'selector': '#search-button', 'input': None},
                    {'action': 'Verify search results', 'selector': '.search-results', 'input': None}
                ]
            }
        }
    
    def identify_test_scenarios(self, requirement_text: str) -> List[str]:
        """Identify test scenarios from requirement text"""
        scenarios = []
        text_lower = requirement_text.lower()
        
        # Pattern matching for common scenarios
        if any(word in text_lower for word in ['login', 'sign in', 'authentication']):
            scenarios.append('login')
        
        if any(word in text_lower for word in ['register', 'sign up', 'registration']):
            scenarios.append('register')
        
        if any(word in text_lower for word in ['search', 'find', 'query']):
            scenarios.append('search')
        
        if any(word in text_lower for word in ['create', 'add', 'insert']):
            scenarios.append('create')
        
        if any(word in text_lower for word in ['update', 'edit', 'modify']):
            scenarios.append('update')
        
        if any(word in text_lower for word in ['delete', 'remove']):
            scenarios.append('delete')
        
        return scenarios
    
    def generate_test_case(self, scenario: str, requirement_text: str, test_id: str) -> TestCase:
        """Generate a test case for a specific scenario"""
        if scenario not in self.test_patterns:
            # Generate generic test case
            return self._generate_generic_test_case(requirement_text, test_id)
        
        pattern = self.test_patterns[scenario]
        steps = []
        
        for i, step_data in enumerate(pattern['steps'], 1):
            step = TestStep(
                step_number=i,
                action=step_data['action'],
                expected_result=f"Successfully {step_data['action'].lower()}",
                selector=step_data['selector'],
                input_data=step_data['input']
            )
            steps.append(step)
        
        return TestCase(
            test_id=test_id,
            title=pattern['title'],
            description=f"Test case for {scenario} functionality based on requirement: {requirement_text[:100]}...",
            priority="Medium",
            steps=steps,
            preconditions=["Application is accessible", "Test data is available"],
            test_data={"scenario": scenario, "requirement": requirement_text}
        )
    
    def _generate_generic_test_case(self, requirement_text: str, test_id: str) -> TestCase:
        """Generate a generic test case when no specific pattern is found"""
        doc = self.nlp(requirement_text)
        
        # Extract key actions and objects
        actions = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        objects = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        
        steps = []
        if actions:
            for i, action in enumerate(actions[:5], 1):  # Limit to 5 steps
                step = TestStep(
                    step_number=i,
                    action=f"Perform {action} action",
                    expected_result=f"Action {action} completed successfully",
                    selector=f"#{action}-element",
                    input_data=None
                )
                steps.append(step)
        
        return TestCase(
            test_id=test_id,
            title=f"Generic Test for {objects[0] if objects else 'Functionality'}",
            description=f"Automatically generated test case for: {requirement_text}",
            priority="Low",
            steps=steps,
            preconditions=["System is available"],
            test_data={"requirement": requirement_text}
        )

def generate_test_cases(requirement_text: str, output_path: str = 'data/generated_tests.json') -> List[TestCase]:
    """
    Main function to generate test cases from requirement text
    
    Args:
        requirement_text: Raw requirement document text
        output_path: Path to save generated test cases as JSON
    
    Returns:
        List of generated TestCase objects
    """
    logger.info("Starting test case generation...")
    
    generator = TestCaseGenerator()
    
    # Split requirements into individual statements
    requirements = [req.strip() for req in requirement_text.split('\n') if req.strip()]
    
    test_cases = []
    test_counter = 1
    
    for requirement in requirements:
        # Identify test scenarios for this requirement
        scenarios = generator.identify_test_scenarios(requirement)
        
        if scenarios:
            for scenario in scenarios:
                test_case = generator.generate_test_case(
                    scenario, requirement, f"TC_{test_counter:03d}"
                )
                test_cases.append(test_case)
                test_counter += 1
        else:
            # Generate generic test case if no specific scenario is identified
            test_case = generator.generate_test_case(
                "generic", requirement, f"TC_{test_counter:03d}"
            )
            test_cases.append(test_case)
            test_counter += 1
    
    # Convert to JSON-serializable format
    test_cases_json = []
    for tc in test_cases:
        tc_dict = asdict(tc)
        # Convert TestStep objects to dictionaries
        tc_dict['steps'] = [asdict(step) for step in tc.steps]
        test_cases_json.append(tc_dict)
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(test_cases_json, f, indent=2, default=str)
    
    logger.info(f"Generated {len(test_cases)} test cases -> {output_path}")
    return test_cases

def extract_actions(requirement_text: str) -> List[Dict]:
    """
    Legacy function for backward compatibility
    """
    test_cases = generate_test_cases(requirement_text)
    return [asdict(tc) for tc in test_cases]

def generate_test_json(requirement_text: str, out_path='data/generated_tests.json'):
    """
    Legacy function for backward compatibility
    """
    generate_test_cases(requirement_text, out_path)

if __name__ == '__main__':
    # Sample requirement document for testing
    sample_requirements = """
    The system shall allow users to login using their email and password.
    The system shall provide user registration functionality with email verification.
    The system shall allow users to search for products by name or category.
    The system shall enable users to add items to their shopping cart.
    The system shall allow users to update their profile information.
    The system shall provide password reset functionality via email.
    """
    
    test_cases = generate_test_cases(sample_requirements)
    print(f"Generated {len(test_cases)} test cases successfully!")
    
    # Print first test case as example
    if test_cases:
        first_tc = test_cases[0]
        print(f"\nExample Test Case:")
        print(f"ID: {first_tc.test_id}")
        print(f"Title: {first_tc.title}")
        print(f"Steps: {len(first_tc.steps)}")
        for step in first_tc.steps:
            print(f"  {step.step_number}. {step.action}")
