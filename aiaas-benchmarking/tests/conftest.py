"""Make the package scripts importable from the tests.

compare_results.py / cost_model.py live in aiaas-benchmarking/ (the parent of
this tests/ dir), so put that on sys.path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
