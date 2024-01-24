import unittest
import numpy as np
from layer import Layer 
from activation import ReLU

class TestLayer(unittest.TestCase):
    def setUp(self):
        self.input_size = 2
        self.output_size = 3
        self.activation = ReLU()
        self.layer = Layer(self.input_size, self.output_size, self.activation)
        self.layer.W = np.array([[10, 20, 30],
                                 [40, 50, 60]])
        self.layer.b = np.ones((self.output_size, 1))

    def test_call(self):
        x = np.array([[-10, 0, 1], 
                      [2, 3, 4]])
        
        self.layer(x)
        expected_z = np.array([[ -19.,   121.,  171.],
                                [ -99.,  151.,  221.],
                                [-179.,  181.,  271.]])
        
        expected_a = np.array([[0., 121., 171.],
                               [0., 151., 221.],
                               [0., 181., 271.,]])

        self.assertEqual(self.layer.x.tolist(), x.tolist())
        self.assertEqual(self.layer.z.tolist(), expected_z.tolist())
        self.assertEqual(self.layer.a.tolist(), expected_a.tolist())

if __name__ == '__main__':
    unittest.main()
