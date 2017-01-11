#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TestMca.py: Tests for the `mcarader` module.
"""

import unittest

import mcareader as mca
import numpy as np


class TestMca(unittest.TestCase):
    def test_mca(self):
        f = mca.Mca("demo.mca")

        xx, yy = f.get_points(trim_zeros=False)
        self.assertEqual(True, isinstance(xx, np.ndarray))
        self.assertEqual(True, isinstance(yy, np.ndarray))
        self.assertEqual(xx.shape, yy.shape)

        self.assertNotEqual("", f.get_section("DATA"))
        self.assertEqual("", f.get_section("UNEXISTING_SECTION"))

        self.assertAlmostEqual(100.0, float(f.get_variable("REAL_TIME")))  # Fixed value in the demo file
        self.assertEqual("", f.get_variable("UNEXISTING_VARIABLE"))


if __name__ == "__main__":
    unittest.main()
