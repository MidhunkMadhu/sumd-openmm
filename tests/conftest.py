import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# run the tests against src/ without requiring `pip install -e .`
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src")))
sys.path.insert(0, HERE)
