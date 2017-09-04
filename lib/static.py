import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
import pywt
import scipy.fftpack
import random
import operator
import copy
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from lib.graph_signal_proc import *
from lib.optimal_cut import *


class Fourier(object):
    """
            Graph Fourier transform.
    """

    def name(self):
        return "FT"

    def set_graph(self, _G):
        self.G = _G
        L = networkx.laplacian_matrix(self.G)
        L = L.todense()
        self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)

    def transform(self, F):
        """
        """
        return graph_fourier(F, self.U)

    def inverse(self, ftr):
        """
        """
        return graph_fourier_inverse(ftr, self.U)

    def drop_frequency(self, ftr, n):
        """
                Keeps only the n top-energy coefficients of ftr.
                Input:
                        * ftr: transform
                        * n: number of coefficients
                Output:
                        * ftr_copy: changed transform
        """
        coeffs = {}

        for i in range(ftr.shape[0]):
            coeffs[i] = abs(ftr[i])

        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

        ftr_copy = numpy.copy(ftr)

        for k in range(n, len(sorted_coeffs)):
            i = sorted_coeffs[k][0]

            ftr_copy[i] = 0

        return ftr_copy


class HWavelets(object):
    """
            Hammond's wavelets (spectral theory)
    """

    def name(self):
        return "HWT"

    def set_graph(self, _G):
        """
        """
        self.G = _G
        L = networkx.normalized_laplacian_matrix(self.G)
        L = L.todense()
        self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)
        lamb_max = max(self.lamb_str.real)

        # default parameters defined by author
        K = 100
        J = 4
        gamma = comp_gamma()
        self.T = comp_scales(lamb_max, K, J)
        self.w = graph_wavelets(self.lamb_str.real, self.U.real,
                                len(self.G.nodes()), self.T)
        self.s = graph_low_pass(self.lamb_str.real, self.U.real,
                                self.T, gamma, lamb_max, K)

    def transform(self, F):
        """
        """
        return hammond_wavelet_transform(self.w, self.s, self.T, F)

    def inverse(self, wtr):
        """
        """
        return hammond_wavelets_inverse(self.w, self.s, wtr)

    def drop_frequency(self, wtr, n):
        """
                Keeps only the n top-energy coefficients of wtr.
                Input:
                        * wtr: transform
                        * n: number of coefficients
                Output:
                        * wtr_copy: changed transform
        """
        coeffs = {}
        for i in range(len(wtr)):
            for j in range(len(wtr[i])):
                coeffs[(i, j)] = abs(wtr[i][j])

        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

        wtr_copy = numpy.copy(wtr)

        for k in range(n, len(sorted_coeffs)):
            i = sorted_coeffs[k][0][0]
            j = sorted_coeffs[k][0][1]

            wtr_copy[i][j] = 0.0

        return wtr_copy


class GRCWavelets(object):
    """
            Gavish's wavelet transform.
    """

    def name(self):
        return "GWT"

    def set_graph(self, _G):
        """
        """
        self.G = _G
        (self.tree, self.ind) = ratio_cut_hierarchy(self.G)

    def transform(self, F):
        """
        """
        return gavish_wavelet_transform(self.tree, self.ind, self.G, F)

    def inverse(self, wtr):
        """
        """
        return gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

    def drop_frequency(self, wtr, n):
        """
                Keeps only the n top-energy coefficients of wtr.
                Input:
                        * wtr: transform
                        * n: number of coefficients
                Output:
                        * wtr_copy: changed transform
        """
        coeffs = {}
        for i in range(len(wtr)):
            coeffs[i] = abs(wtr[i])

        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

        wtr_copy = numpy.copy(wtr)

        for k in range(n, len(sorted_coeffs)):
            i = sorted_coeffs[k][0]

            wtr_copy[i] = 0.0

        return wtr_copy


class OptWavelets(object):
    """
            Sparse wavelet transform.
    """

    def __init__(self, n=0):
        """
        """
        self.n = n

    def name(self):
        """
        """
        if self.n == 0:
            return "SWT"
        else:
            return "FSWT"

    def set_graph(self, _G):
        self.G = _G

    def transform(self, F):
        """
        """
        self.F = F
        return None

    def inverse(self, wtr):
        """
        """
        return gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

    def drop_frequency(self, wtr, n):
        coeffs = {}

        k = n
        # Computing optimal basis
        (self.tree, self.ind, s) = optimal_wavelet_basis(self.G, self.F, k, self.n)

        # Gavish's wavelet transform
        tr = gavish_wavelet_transform(self.tree, self.ind, self.G, self.F)

        for i in range(len(tr)):
            coeffs[i] = abs(tr[i])

        sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

        wtr_copy = numpy.copy(tr)

        # Computing number of integers required to represent the edges cut (rounded)
        v = n - int(math.ceil(float(s * math.log2(len(self.G.edges()))) / 64))

        for k in range(v, len(sorted_coeffs)):
            i = sorted_coeffs[k][0]

            wtr_copy[i] = 0.0

        return wtr_copy
