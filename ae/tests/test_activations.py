import unittest 
import numpy as np
from src.activation import ReLU


class TestReLU(unittest.TestCase):
    def test_call(self):
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        a = relu(x) 
        expected_a = np.array([0, 0, 0, 1, 2])
        np.testing.assert_equal(a, expected_a)

    def test_compute_grad(self):
        relu = ReLU()
        relu(np.array([-2, -1, 0, 1, 2]))
        grad = relu.compute_grad()
        expected_a = np.array([0, 0, 0, 1, 1])
        np.testing.assert_equal(grad, expected_a)


if __name__ == '__main__':
    unittest.main()
