"""A minimal python interface to read Amptek's mca files"""

import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import linregress

from io import open  # python 2 compatibility

__author__ = 'Dih5'
__version__ = "0.2.0"


def _str_to_array(s, sep=" "):
    """
    Convert a string to a numpy array

    Args:
        s (str): The string
        sep (str): The separator in the string.

    Returns:
        (`numpy.ndarray`): The 2D array with the data.

    """
    line_len = len(np.fromstring(s[:s.find('\n')], sep=sep))
    return np.reshape(np.fromstring(s, sep=sep), (-1, line_len))


class Mca:
    """
    A mca file.

    Attributes:
        raw(str): The text of the file.

    """

    def __init__(self, file, encoding='iso-8859-15'):
        """
        Load the content of a mca file.

        Args:
            file(str): The path to the file.
            encoding (str): The encoding to use to read the file.


        """
        with open(file, "r", encoding=encoding) as f:
            self.raw = f.read()

    def get_section(self, section):
        """
        Find the str representing a section in the MCA file.

        Args:
            section(str): The name of the section to search for.

        Returns:
            (str): The text of the section or "" if not found.

        """
        m = re.match(r"(?:.*)(^<<%s>>$)(.*?)(?:<<.*>>)" % section, self.raw, re.DOTALL + re.MULTILINE)
        return m.group(2).strip() if m else ""

    def get_variable(self, variable):
        """
        Find the str representing a variable in the MCA file.

        Args:
            variable(str): The name of the variable to search for.

        Returns:
            (str): The text of the value or "" if not found.

        """
        m = re.match(r"(?:.*)%s - (.*?)$" % variable, self.raw, re.DOTALL + re.MULTILINE)
        return m.group(1).strip() if m else ""

    def get_calibration_points(self):
        """
        Get the calibration points from the MCA file.


        Returns:
            (`numpy.ndarray`): The 2D array with the data.

        """
        cal = self.get_section("CALIBRATION")
        cal = cal[cal.find('\n') + 1:]  # remove first line
        return _str_to_array(cal)

    def get_calibration_function(self, method=None):
        """
        Get a calibration function from the file.

        Args:
            method(str): The method to use. Available methods include:

                * 'interpolation': Use a linear interpolation.
                * 'bestfit': A linear fit in the sense of least-squares.


        Returns:
            (`Callable`): A function mapping channel number to energy.

        """
        points = self.get_calibration_points()
        info = sys.version_info
        if info[0] == 3 and info[1] < 4 or info[0] == 2 and info[1] < 7: #py2 < 2.7 or py3 < 3.4
            extrapolation_support = False
        else:
            extrapolation_support = True

        if method is None:
            method = "bestfit"

        if method == "interpolation" and not extrapolation_support:
            warnings.warn("Warning: extrapolation not supported with active Python interpreter. Using best fit instead")
            method = "bestfit"

        if method == "interpolation":
            return interpolate.interp1d(points[:, 0], points[:, 1], fill_value="extrapolate")
        elif method == "bestfit":
            slope, intercept, _, _, _ = linregress(points[:, 0], points[:, 1])
            return np.vectorize(lambda x: slope * x + intercept)
        else:
            raise ValueError("Unknown method: %s" % method)

    def get_points(self, calibration_method=None, trim_zeros=True, background=None):
        """
        Get the points of the spectrum.

        Args:
            calibration_method (str): The method used for the calibration. See `get_calibration_function`.
            trim_zeros (bool): Whether to remove values with no counts.
            background (`Mca`): An spectrum describing a background to subtract from the returned points. The background
                                is scaled using the REAL_TIME parameters.

        Returns:
            (tuple): tuple containing:

                x (List[float]): The list of x coordinates (mean bin energy).

                y (List[float]): The list of y coordinates (counts in each bin).

        """

        f = self.get_calibration_function(method=calibration_method)
        yy = _str_to_array(self.get_section("DATA"))[:, 0]
        if background:
            background_yy = _str_to_array(background.get_section("DATA"))[:, 0]
            yy -= background_yy * (float(self.get_variable("REAL_TIME")) / float(background.get_variable("REAL_TIME")))
        xx = f(range(len(yy)))
        if trim_zeros:
            yy = np.trim_zeros(yy, 'f')
            xx = xx[len(xx) - len(yy):]
            yy = np.trim_zeros(yy, 'b')
            removed_count = len(yy) - len(xx)
            if removed_count:  # Then removed_count is negative
                xx = xx[:len(yy) - len(xx)]
        return xx, yy

    def plot(self, log_y=False, log_x=False, calibration_method=None):
        """
        Show a plot of the spectrum.

        Args:
            log_y(bool): Whether the y-axis is in logarithmic scale.
            log_x(bool): Whether the x-axis is in logarithmic scale.
            calibration_method (str): The method used for the calibration. See `get_calibration_function`.

        """
        xx, yy = self.get_points(calibration_method=calibration_method)
        if log_y and log_x:
            plt.loglog(xx, yy)
        elif log_y:
            plt.semilogy(xx, yy)
        elif log_x:
            plt.semilogx(xx, yy)
        else:
            plt.plot(xx, yy)
        plt.show()
