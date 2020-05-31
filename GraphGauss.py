import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

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
    sigma = fwhm/(2 * math.sqrt(2 * math.log(2)))
    return sigma

def make_gaussian(center=0, fwhm=1, max=1):
    sigma = variance_from_FWHM(fwhm)
    x = np.linspace(center - 3 * sigma, center + 3 * sigma, 100)
    val = stats.norm.pdf(x, center, sigma)
    factor = max / val[50]
    return [x, center, sigma, factor, fwhm, max]

def graph_gaussians(gaussian_list):
    plt.title('Gaussian Distribution')
    count = 0
    colors = ['r', 'b', 'g', 'y', 'o']
    for g in gaussian_list:
        label = 'center: {} fwhm: {} max: {}'.format(g[1], g[4], g[5])
        plt.plot(g[0], stats.norm.pdf(g[0], g[1], g[2]) * g[3], colors[count], label=label)
        count += 1
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()

def main():
    example1 = make_gaussian(center=1, fwhm=2, max=10)
    example2 = make_gaussian(center=4, fwhm=3, max=30)
    gaussians = [example1, example2]
    graph_gaussians(gaussians)


if __name__ == '__main__':
    main()
