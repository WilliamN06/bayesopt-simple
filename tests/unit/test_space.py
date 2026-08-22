"""Unit tests for ParameterSpace."""

import pytest
import numpy as np
from bayesopt.core.space import ParameterSpace, ParameterType


class TestParameterSpace:
    """Test ParameterSpace class."""
    
    def test_continuous_parameter(self):
        bounds = {'x': (0.0, 1.0)}
        space = ParameterSpace(bounds)
        
        assert space.n_params == 1
        assert space.param_names == ['x']
        assert space.param_types['x'] == ParameterType.CONTINUOUS
        assert space.param_ranges['x'] == (0.0, 1.0)
        
        sample = space.sample_random()
        assert 0.0 <= sample['x'] <= 1.0
        
        arr = space.to_array(sample)
        recovered = space.from_array(arr)
        assert recovered['x'] == pytest.approx(sample['x'])
    
    def test_integer_parameter(self):
        bounds = {'n': (0, 10, 'int')}
        space = ParameterSpace(bounds)
        
        assert space.param_types['n'] == ParameterType.INTEGER
        assert space.param_ranges['n'] == (0, 10)
        
        for _ in range(100):
            sample = space.sample_random()
            assert isinstance(sample['n'], int)
            assert 0 <= sample['n'] <= 10
        
        sample = {'n': 7}
        arr = space.to_array(sample)
        recovered = space.from_array(arr)
        assert recovered['n'] == 7
    
    def test_log_parameter(self):
        bounds = {'lr': (1e-4, 1e-1, 'log')}
        space = ParameterSpace(bounds)
        
        assert space.param_types['lr'] == ParameterType.LOG
        assert space.param_ranges['lr'] == (1e-4, 1e-1)
        
        for _ in range(100):
            sample = space.sample_random()
            assert 1e-4 <= sample['lr'] <= 1e-1
        
        sample = {'lr': 1e-3}
        arr = space.to_array(sample)
        recovered = space.from_array(arr)
        assert recovered['lr'] == pytest.approx(sample['lr'])
    
    def test_categorical_parameter(self):
        bounds = {'act': ('relu', 'tanh', 'sigmoid')}
        space = ParameterSpace(bounds)
        
        assert space.param_types['act'] == ParameterType.CATEGORICAL
        assert space.categorical_mappings['act'] == ['relu', 'tanh', 'sigmoid']
        
        samples = set()
        for _ in range(100):
            sample = space.sample_random()
            assert sample['act'] in ['relu', 'tanh', 'sigmoid']
            samples.add(sample['act'])
        assert len(samples) == 3
        
        sample = {'act': 'tanh'}
        arr = space.to_array(sample)
        recovered = space.from_array(arr)
        assert recovered['act'] == 'tanh'
    
    def test_mixed_parameters(self):
        bounds = {
            'lr': (1e-4, 1e-1, 'log'),
            'n': (50, 500, 'int'),
            'act': ('relu', 'tanh', 'sigmoid'),
            'rate': (0.0, 1.0)
        }
        space = ParameterSpace(bounds)
        
        assert space.n_params == 4
        assert space.param_names == ['lr', 'n', 'act', 'rate']
        
        sample = space.sample_random()
        assert 'lr' in sample
        assert 'n' in sample
        assert 'act' in sample
        assert 'rate' in sample
        
        arr = space.to_array(sample)
        recovered = space.from_array(arr)
        assert recovered['lr'] == pytest.approx(sample['lr'])
        assert recovered['n'] == sample['n']
        assert recovered['act'] == sample['act']
        assert recovered['rate'] == pytest.approx(sample['rate'])
    
    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            ParameterSpace({})
        
        with pytest.raises(ValueError):
            ParameterSpace({'x': (1.0, 0.0)})
        
        with pytest.raises(ValueError):
            ParameterSpace({'x': (0.0, 1.0, 'log')})
        
        with pytest.raises(ValueError):
            ParameterSpace({'x': (1.0, 0.0, 'log')})
        
        with pytest.raises(ValueError):
            ParameterSpace({'x': ('a', 'b', 'a')})
    
    def test_bounds_array(self):
        bounds = {
            'x': (0.0, 1.0),
            'y': (0.0, 10.0, 'int'),
            'z': ('a', 'b', 'c')
        }
        space = ParameterSpace(bounds)
        bounds_arr = space.bounds_array()
        
        assert bounds_arr.shape == (3, 2)
        assert bounds_arr[0, 0] == 0.0
        assert bounds_arr[0, 1] == 1.0
        assert bounds_arr[1, 0] == 0.0
        assert bounds_arr[1, 1] == 1.0
        assert bounds_arr[2, 0] == 0.0
        assert bounds_arr[2, 1] == 1.0
    
    def test_get_default(self):
        bounds = {
            'x': (0.0, 1.0),
            'n': (0, 10, 'int'),
            'act': ('a', 'b', 'c', 'd')
        }
        space = ParameterSpace(bounds)
        default = space.get_default()
        
        assert default['x'] == pytest.approx(0.5)
        assert default['n'] == 5
        assert default['act'] in ['a', 'b', 'c', 'd']
    
    def test_multiple_samples(self):
        bounds = {'x': (0.0, 1.0)}
        space = ParameterSpace(bounds)
        
        samples = space.sample_random(n_points=10)
        assert len(samples) == 10
        for sample in samples:
            assert 0.0 <= sample['x'] <= 1.0
    
    def test_unknown_parameter(self):
        bounds = {'x': (0.0, 1.0)}
        space = ParameterSpace(bounds)
        
        with pytest.raises(KeyError):
            space.to_array({'y': 0.5})
    
    def test_categorical_invalid_value(self):
        bounds = {'act': ('relu', 'tanh')}
        space = ParameterSpace(bounds)
        
        with pytest.raises(ValueError):
            space.to_array({'act': 'sigmoid'})