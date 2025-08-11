"""
Drop Allocator Package

A Python-based optimization system for allocating drops with coverage encouragement.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .main import main, greedy, ilp_ortools, ilp_pulp
from .coverage import *

__all__ = [
    "main",
    "greedy", 
    "ilp_ortools",
    "ilp_pulp",
]
