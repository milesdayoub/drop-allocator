"""
Tests for the allocator module
"""

import pytest
import pandas as pd
import numpy as np
from src.allocator.main import greedy, summarize


class TestAllocator:
    """Test cases for the allocator module"""
    
    def test_greedy_basic(self):
        """Test basic greedy allocation"""
        # Create test data
        caps = pd.DataFrame({
            'contract_address': ['A', 'B', 'C'],
            'cap_face': [10, 10, 10],
            'is_sponsored': [True, True, False]
        })
        
        elig = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U2'],
            'contract_address': ['A', 'B', 'C', 'A', 'B', 'C'],
            'score': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        })
        
        result = greedy(caps, elig, k=2, seed=42)
        
        assert not result.empty
        assert len(result) <= 4  # 2 users * 2 allocations max
        assert 'user_id' in result.columns
        assert 'contract_address' in result.columns
    
    def test_summarize(self):
        """Test summary statistics"""
        # Create test assignment data
        df_assign = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U2', 'U2', 'U3'],
            'contract_address': ['A', 'B', 'A', 'C', 'B']
        })
        
        elig = pd.DataFrame({
            'user_id': ['U1', 'U2', 'U3'],
            'contract_address': ['A', 'A', 'B'],
            'score': [0.9, 0.8, 0.7]
        })
        
        summary = summarize(df_assign, elig, k=2)
        
        assert 'n_users' in summary
        assert 'dist' in summary
        assert 'fill_rate' in summary
        assert summary['n_users'] == 3
        assert isinstance(summary['dist'], dict)
        assert 0 <= summary['fill_rate'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
