import numpy as np
from scipy import stats
from scipy import signal
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pylab as plt
import math

def stdevToFWHM(stdev):
    """
    Function converts sigma, or the standard deviation, to the Gaussian's
    Full Width Half Maximum (FWHM). The formula is from Wikipedia (link:
    https://en.wikipedia.org/wiki/Full_width_at_half_maximum).
    Only works for symmetric curves.
    :param stdev: Standard Deviation / Sigma
    :return: FWHM
    """
    FWHM = 2 * np.sqrt(2 * np.log(2)) * stdev
    return FWHM

def createGauss(center, width):
    """
    Function takes the center and the width (FWHM, or Full Width at Half Maximum)
    and generates a Gaussian distribution. gauss computes from stats.norm, and
    gauss1 computes from a formula from Wikipedia (link:
    https://en.wikipedia.org/wiki/Gaussian_function)
    :param center: the x-axis location of the Gaussian peak (for a symmetric curve)
    :param width: the FWHM
    :param x: an array of points around the center
    :return:
    """
    gauss = stats.norm(center,width)
    #gauss2 = math.exp(-4*np.log(2)*((x-center)^2)/(width)^2)
    return gauss


def computeGauss(intensity, fwhm):
    result = np.convolve(intensity, fwhm)
    return result

def gaussianProfile(x, fwhm):
    """
    This function is a python translation of the GaussianProfile function
    found in GaussianProfile.cpp in diffpy/libdiffpy.
    :param x: variable
    :param fwhm: FWHM, or Full Width at Half Maximum of the Gaussian
    :return: Gaussian profile
    """
    if fwhm<=0:
        return 0.0
    xrel = x/fwhm
    rv = 2 * math.sqrt(np.log(2) / math.pi) / fwhm * math.exp(-4 * np.log(2) * xrel * xrel)
    return rv


print(createGauss(2,7))
print(computeGauss(2,6))
print(gaussianProfile(2, 6))

