#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Virtual environment set up. Activate with 'source venv/bin/activate'."
