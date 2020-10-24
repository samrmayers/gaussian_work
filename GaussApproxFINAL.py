import math
import pandas as pd
from sklearn import linear_model
import numpy
import GraphGauss as gg
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import sklearn.metrics as sm


def GaussApprox(c, stdev, intensity, ci, xj, ratio):
    """
    This function generates the intensities of the "mini" Gaussians
    that when summed, approximate one full Gaussian.

    :param c: The x-value center of the Gaussian being approximated.
    :param stdev: The standard deviation of the Gaussian being approximated.
    :param intensity: The intensity of the Gaussian being approximated.
    :param ci: the locations of the smaller peaks
    :param xj: an array of "test points"
    :param ratio: the ratio of the fwhm of the smaller curves to the fwhm of the larger one
    :return: An array of the intensities of the "mini" Gaussians.
    """

    # generate two arrays for linear regression
    Y = []
    Z = []
    for x in xj:  # at each x, get an input and output pair
        Y.append(gety(ci, stdev * ratio, x))
        Z.append(getz(c, stdev, intensity, x))

    # multiple linear regression
    regr = linear_model.LinearRegression()
    regr.fit(Y, Z)
    #print('Coefficients: \n', regr.coef_)
    return regr.coef_


def gety(ci, stdev, x):
    y = []
    for i in ci:
        y.append(math.exp((-(x - i) ** 2) / (2 * stdev ** 2)))
    return y


def getz(c, stdev, intensity, x):
    z = intensity * (math.exp((-(x - c) ** 2) / (2 * stdev ** 2)))
    return z


def stdevToFWHM(stdev):
    FWHM = 2 * numpy.sqrt(2 * numpy.log(2)) * stdev
    return FWHM


def generate_gaussians_varieddist(c, stdev, intensity, smallest_stdev):
    """
    This function takes in the information for a Gaussian and generates 5 miniature Gaussians
    at various distances from the original Gaussians center that when summed, approximate the original Gaussian.

    :param c: The x-value center of the Gaussian being approximated.
    :param stdev: The standard deviation of the Gaussian being approximated.
    :param intensity: The intensity of the Gaussian being approximated.
    :param smallest_stdev: The standard deviation of the smallest Gaussian in a larger set of Gaussians
    all requiring approximations. This smallest standard deviation will be used as the standard deviation
    for all "mini" Gaussians generated to approximate the larger Gaussian.

    :return: a list of Gaussians (their information), including the original gaussian and the 5 mini gaussians,
    a single Gaussian which is the sum of the mini Gaussians (the final approximation of the original curve -
    this one is the actual list of y values, not the list of information -- see GraphGauss.make_gaussian),
    and the mean absolute error (between final approximation and original Gaussian)
    """
    gaussians = []
    n = 1.25 # this number can be changed, used to vary the distance of the test points and loc of mini gaussians (see below)


    xj = [c - (3 / n) * stdev, c - (2.5 / n) * stdev, c - (2 / n) * stdev, c - (1 / n) * stdev,
          c - (.5 / n) * stdev, c, c + (.5 / n) * stdev, c + (1 / n) * stdev, c + (2 / n) * stdev,
          c + (2.5 / n) * stdev]
    ci = [c - (2 / n) * stdev, c - (1 / n) * stdev, c, c + (1 / n) * stdev, c + (2 / n) * stdev]

    gaussians.append(gg.make_gaussian(center=c, fwhm=stdevToFWHM(stdev), max=intensity))
    ratio = smallest_stdev / stdev
    intensities = GaussApprox(c, stdev, intensity, ci, xj, ratio)

    for i in range(0, len(ci)):
        gaussians.append(gg.make_gaussian(center=ci[i], fwhm=stdevToFWHM(stdev * ratio), max=intensities[i]))

    # get the sum of the smaller Gaussians
    approx_gauss = 0
    for i in range(1,len(gaussians)):
        g = gaussians[i]
        gauss = stats.norm.pdf(gaussians[0][0], g[1], g[2]) * g[3]
        approx_gauss = approx_gauss + gauss

    # get error here
    mae = sm.mean_absolute_error(approx_gauss, (stats.norm.pdf(gaussians[0][0], gaussians[0][1],
                                                         gaussians[0][2]) * gaussians[0][3]))

    return gaussians, approx_gauss, mae


def generate_gaussians_variednum(c, stdev, intensity, smallest_stdev):
    """
    This function takes in the information for a Gaussian and generates various numbers of miniature
    Gaussians at set distances that when summed, approximate the original Gaussian.

    :param c: The x-value center of the Gaussian being approximated.
    :param stdev: The standard deviation of the Gaussian being approximated.
    :param intensity: The intensity of the Gaussian being approximated.
    :param smallest_stdev: The standard deviation of the smallest Gaussian in a larger set of Gaussians
    all requiring approximations. This smallest standard deviation will be used as the standard deviation
    for all "mini" Gaussians generated to approximate the larger Gaussian.

    :return: a list of Gaussians (their information), including the original gaussian and the mini gaussians,
    a single Gaussian which is the sum of the mini Gaussians (the final approximation of the original curve -
    this one is the actual list of y values, not the list of information -- see GraphGauss.make_gaussian),
    and the mean absolute error (between final approximation and original Gaussian)
    """
    gaussians = []
    n = 1.25 # this number can be changed, used to vary the distance of the test points and loc of mini gaussians (see below)
    xj = [c - (3 / n) * stdev, c - (2.5 / n) * stdev, c - (2 / n) * stdev, c - (1 / n) * stdev,
          c - (.5 / n) * stdev, c, c + (.5 / n) * stdev, c + (1 / n) * stdev, c + (2 / n) * stdev,
          c + (2.5 / n) * stdev]

    if smallest_stdev/stdev > .6: # this number taken from Coelho paper
        num_gaussians = 5
        n = 1.25
    else:
        num_gaussians = 10
        n = 2
    ci = []
    ci.append(c)
    for i in range(1,math.ceil(num_gaussians/2)):
        ci.append(c - (i / n) * stdev)
        ci.append((c + (i / n) * stdev))
    ci.sort()

    gaussians.append(gg.make_gaussian(center=c, fwhm=stdevToFWHM(stdev), max=intensity))
    ratio = smallest_stdev / stdev
    intensities = GaussApprox(c, stdev, intensity, ci, xj, ratio)

    for i in range(0, len(ci)):
        gaussians.append(gg.make_gaussian(center=ci[i], fwhm=stdevToFWHM(stdev * ratio), max=intensities[i]))

    # get the sum of the smaller Gaussians
    approx_gauss = 0
    for i in range(1,len(gaussians)):
        g = gaussians[i]
        gauss = stats.norm.pdf(gaussians[0][0], g[1], g[2]) * g[3]
        approx_gauss = approx_gauss + gauss

    # get error here
    mae = sm.mean_absolute_error(approx_gauss, (stats.norm.pdf(gaussians[0][0], gaussians[0][1],
                                                         gaussians[0][2]) * gaussians[0][3]))

    return gaussians, approx_gauss, mae


def main():
    gaussians = []
    total_approxes = []
    maes = []
    ogs = []
    # the smallest sdtev is 1, so that is where the ratios will come from
    smallest_stdev = 1

    # generate 10 Gaussians
    c = numpy.arange(0, 100, 10)
    stdev = numpy.arange(1, 3, .2)
    intensity = []
    for i in range(0, len(c)):
        intensity.append(random.randint(1, 5))

    # TEST VARIED DISTANCES
    """for n in range(0, len(c)):
        if n < len(c) / 2 - 1:
            approxs, total_approx, mae = generate_gaussians_varieddist(c[n], stdev[n], intensity[n], smallest_stdev)
            gaussians = gaussians + approxs
            total_approxes.append(total_approx)
            maes.append(mae)

        else:
            smallest_stdev = stdev[4]
            approxs, total_approx, mae = generate_gaussians_varieddist(c[n], stdev[n], intensity[n], smallest_stdev)
            gaussians = gaussians + approxs
            total_approxes.append(total_approx)
            maes.append(mae)
        #approxs, total_approx, mae = generate_gaussians_varieddist(c[n], stdev[n], intensity[n], smallest_stdev)
        #gaussians = gaussians + approxs
        #total_approxes.append(total_approx)
        #maes.append(mae)
    gg.graph_gaussians(gaussians,maes)
    #show the approximation for one curve -- the seventh curve
    gg.graph_gausswapprox(gaussians[36:42])"""


    # TEST VARIED NUMBER
    """for n in range(0,len(c)):
        approxs, total_approx, mae = generate_gaussians_variednum(c[n], stdev[n], intensity[n], smallest_stdev)
        gaussians = gaussians + approxs[1:]
        total_approxes.append(total_approx)
        maes.append(mae)
        ogs.append(approxs[0])
    gg.graph_gaussians_varied(gaussians,maes,ogs)"""

    # INDIVIDUAL GAUSS APPROX - VARIED NUM
    approxs, total_approx, mae = generate_gaussians_variednum(c[8], stdev[8], intensity[8], smallest_stdev)
    gg.graph_gausswapprox(approxs)



if __name__ == '__main__':
    main()