#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TestMca.py: Tests for the `mcareader` module.
"""

import unittest

import numpy as np
import os

import mcareader as mca


class TestMca(unittest.TestCase):
    def test_points(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demo.mca"))

        for method in ["bestfit", "interpolation"]:
            for trim_zeros in [True, False]:
                xx, yy = f.get_points(trim_zeros=trim_zeros, calibration_method=method)
                self.assertTrue(isinstance(xx, np.ndarray))
                self.assertTrue(isinstance(yy, np.ndarray))
                self.assertEqual(xx.shape, yy.shape)

    def test_variables(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demo.mca"))
        # Check the fixed values in the demo file
        # Check all the patterns in the file
        self.assertEqual("3", f.get_variable("GAIN"))
        self.assertEqual("80", f.get_variable("CLCK"))
        self.assertEqual("217K", f.get_variable("TEC Temp"))
        self.assertEqual("", f.get_variable("UNEXISTING_VARIABLE"))

    def test_sections(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demo.mca"))
        self.assertNotEqual("", f.get_section("DATA"))
        self.assertEqual("", f.get_section("UNEXISTING_SECTION"))

    def test_background(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demo.mca"))
        f2 = mca.Mca(os.path.join(os.path.dirname(__file__), "demo2.mca"))
        # demo2 has the same counts in double time. Hence, subtraction must yield half of the original counts

        xx, yy = f.get_points()
        xx2, yy2 = f.get_points(background=f2)
        for y, y2 in zip(yy, yy2):
            self.assertAlmostEqual(y, y2*2)

    def test_fit(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demo.mca"))
        g = f.get_calibration_function(method="bestfit")
        # Values calculated independently
        self.assertAlmostEqual(g(0), 1./6.)
        self.assertAlmostEqual(g(1)-g(0), 0.05)

    def test_counts(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demo.mca"))
        # Values calculated independently using a spreadsheet
        self.assertAlmostEqual(f.get_counts(), 96897)
        self.assertAlmostEqual(f.get_total_energy(), 978116.75)

    def test_no_calibration(self):
        f = mca.Mca(os.path.join(os.path.dirname(__file__), "demoNoCal.mca"))
        self.assertIsNone(f.get_calibration_points())
        g = f.get_calibration_function()
        # Check g is the identity
        for x in [3.0, 7.2, 103.5]:
            self.assertEqual(g(x), x)

if __name__ == "__main__":
    unittest.main()
