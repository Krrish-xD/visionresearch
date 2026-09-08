"""
Deprecated / Backwards-Compatibility Wrapper.
Redirects to scripts/setup_resources.py.
"""

import sys
import os

# Redirect directly to setup_resources.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.setup_resources import main

if __name__ == "__main__":
    main()
