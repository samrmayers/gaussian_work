import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import sklearn.metrics as sm


def make_normal(m, variance):
    mu = m
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    return x, mu, sigma


def make_normal_with_sdev(mu, sigma):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    return x, mu, sigma


def make_graph(x, mu, sigma):
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.title('Gaussian Distribution')
    plt.show()


def variance_from_FWHM(fwhm):
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    return sigma


def make_gaussian(center=0, fwhm=1, max=1):
    sigma = variance_from_FWHM(fwhm)
    x = np.linspace(center - 3 * sigma, center + 3 * sigma, 100)
    val = stats.norm.pdf(x, center, sigma)
    factor = max / val[50]
    return [x, center, sigma, factor, fwhm, max]


def graph_gaussians(gaussian_list,maes):
    plt.title('Gaussian Distribution')
    count = 0
    count2 = 0
    colors = ['r', 'b', 'b', 'b', 'b', 'b']
    linewidths = [3, 1, 1, 1, 1, 1]
    for g in gaussian_list:
        if linewidths[count] == 3:
            plt.plot(g[0], stats.norm.pdf(g[0], g[1], g[2]) * g[3], colors[count], linewidth=linewidths[count],label=maes[count2])
            count2 = count2 + 1
        else:
            plt.plot(g[0], stats.norm.pdf(g[0], g[1], g[2]) * g[3], colors[count], linewidth=linewidths[count])
        if count == 5:
            count = 0
        else:
            count = count + 1
    plt.grid()
    plt.legend(title = "Mean Absolute Errors",loc='upper left',fontsize = 'small')
    plt.show()


def graph_gausswapprox(gaussian_list):
    plt.title("Gaussian with Approximation")
    approx = 0
    x = gaussian_list[0][0]
    for g in gaussian_list[1:]:
        gauss = stats.norm.pdf(x, g[1], g[2]) * g[3]
        approx = approx + gauss
    plt.plot(x, approx, color='g', label='Approximation',linewidth = 3)
    plt.plot(x, stats.norm.pdf(gaussian_list[0][0], gaussian_list[0][1], gaussian_list[0][2]) * gaussian_list[0][3],
             color='r', label='Original')
    mae = sm.mean_absolute_error(approx,(stats.norm.pdf(gaussian_list[0][0], gaussian_list[0][1],
                                           gaussian_list[0][2]) * gaussian_list[0][3]))
    plt.grid()
    plt.annotate('MAE = {}'.format(mae), xy=(gaussian_list[0][1], .5))
    plt.legend(loc='upper left')
    plt.show()


def graph_gaussians_varied(gaussian_list,maes,ogs):
    plt.title('Gaussian Distribution')
    count2 = 0
    for g in ogs:
        plt.plot(g[0], stats.norm.pdf(g[0], g[1], g[2]) * g[3], 'r', linewidth=3, label=maes[count2])
        count2 = count2 + 1
    for g in gaussian_list:
        plt.plot(g[0], stats.norm.pdf(g[0], g[1], g[2]) * g[3], 'b', linewidth=1)
    plt.grid()
    plt.legend(title = "Mean Absolute Errors",loc='upper left',fontsize = 'small')
    plt.show()


def main():
    example1 = make_gaussian(center=1, fwhm=2, max=10)
    example2 = make_gaussian(center=4, fwhm=3, max=30)
    gaussians = [example1, example2]
    graph_gaussians(gaussians)


if __name__ == '__main__':
    main()
