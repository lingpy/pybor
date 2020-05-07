"""
Utility functions for the package.
"""
from pathlib import Path

def test_data(*comps):
    return Path(__file__).parent.parent.parent.joinpath('tests', 'data',
            *comps).as_posix()
