import math
from collections import deque

import networkx as nx
import numpy as np
from numpy import dot, diag, sqrt
from numpy.linalg import eigh
import scipy.optimize
from scipy import linalg
import scipy.fftpack


def compute_eigenvectors_and_eigenvalues(L):
    """
        Computes eigenvectors and eigenvalues of the matrix L
        Input:
            * L: matrix
        Output:
            * U: eigenvector matrix, one vector/column, sorted by corresponding
                 eigenvalue
            * lamb: eigenvalues, sorted in increasing order
    """
    lamb, U = linalg.eig(L)

    idx = lamb.argsort()
    lamb = lamb[idx]
    U = U[:, idx]

    return U, lamb


def g(x):
    """
        Wavelet generating kernel, see Hammond, D. K.,Vandergheynst, P.,
        & Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory".
        Input:
            * x
        Output:
            * kernel of x
    """
    a = 2
    b = 2
    x_1 = 1
    x_2 = 2

    if x < x_1:
        return pow(x_1, -a) * pow(x, a)
    elif x <= x_2 and x >= x_1:
        return -5 + 11 * x - 6 * pow(x, 2) + pow(x, 3)
    else:
        return pow(x_2, b) * pow(x, -b)


def comp_gamma():
    """
        In Hammond, D. K.,Vandergheynst, P.,& Gribonval, R. (2011).
        "Wavelets on graphs via spectral graph theory" gamma is a parameter
        used to determine the scaling function h. It is such that h(0) = max(g)
        Input:
                * None
        Output:
                * Gamma function (array)
    """
    # fminbound finds the minimum within the optimization bounds
    xopt = scipy.optimize.fminbound(lambda x: -g(x), 1, 2)
    return xopt


def h(x, gamma, lamb_max, K):
    """
        Scaling function see Hammond, D. K.,Vandergheynst, P.,
        & Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory".
        Input:
                * x
                * gamma
                * lamb_max: upper bound spectrum
                * K: normalization
        Output:
                * value of scaling function
    """
    lamb_min = float(lamb_max) / K
    return gamma * math.exp(-pow(float(x / (lamb_min * 0.6)), 4))


def comp_scales(lamb_max, K, J):
    r"""
        Computes wavelet scales see Hammond, D. K.,Vandergheynst, P.,
        & Gribonval, R. (2011). "Wavelets on graphs via spectral graph theory".
        Input:
            * lamb_max: upper bound spectrum
            * K: desired ratio for lambda_max / lambda_min
            * J: number of scales
        Output:
            * scales array
    """
    lamb_min = float(lamb_max) / K
    s_min = float(1) / lamb_max
    s_max = float(2) / lamb_min

    return np.exp(np.linspace(math.log(s_max), math.log(s_min), J))


def graph_low_pass(lamb, U, T, gamma, lamb_max, K):
    """
        Low-pass spectral filter (square matrix).
        See "The emerging field of signal processing on graph"
        Input:
            * lamb: eigenvalues
            * U: eigenvector matrix
            * N: number of nodes
            * T: wavelet scales
            * gamma: scaling function parameter
            * lamb_max: upper-bound spectrum
            * K: normalization
        Output:
            * s: Low-pass filter as a N x N matrix
    """

    N = len(lamb)

    h_vector_form = [h(T[-1] * lamb[x], gamma, lamb_max, K) for x in range(N)]

    s = []

    for n in range(N):
        s.append([])

        for m in range(N):
            s_n_m = 0

            for x in range(N):
                s_n_m = s_n_m + U[n][x] * U[m][x] * h_vector_form[x]

            s[n].append(s_n_m)

    return s


def graph_wavelets(lamb, U, N, T):
    """
        Graph wavelets.
        Input:
            * lamb: eigenvalues
            * U: eigenvector matrix
            * N: number of nodes
            * T: wavelet scales
        Output:
            * w: wavelets as a len(T) x N x N matrix
    """
    w = []

    for t in range(len(T)):
        w.append([])
        for n in range(N):
            w[t].append([])
            for m in range(N):
                w_t_n_m = 0
                for x in range(N):
                    w_t_n_m = w_t_n_m + U[n][x] * U[m][x] * g(T[t] * lamb[x])

                w[t][n].append(w_t_n_m)

    return w


def graph_fourier(F, U):
    """
        Graph Fourier transform.
        Input:
            * F: Signal in the vertex domain
            * U: Eigenvectors matrix
        Ouput:
            * F_hat: Signal in the graph spectral domain
    """
    F_hat = []

    for i in range(0, len(U)):
        F_hat.append(np.dot(F, U[:, i]))

    F_hat = np.array(F_hat)

    return F_hat


def graph_fourier_inverse(F_hat, U):
    """
        Graph Fourier inverse:
        Input:
            * F_hat: Signal in the graph spectral domain
            * U: Eigenvectors matrix
        Output:
            * F: Signal in the vertex domain
    """
    F = np.zeros(U.shape[0])
    for v in range(U.shape[0]):
        for u in range(U.shape[1]):
            F[v] = F[v] + (F_hat[u] * U[v][u]).real

    return F


def hammond_wavelet_transform(w, s, T, F):
    r"""
        Hammond wavelet transform.
        Input:
            * w: wavelets
            * s: low-pass wavelet (scaling function)
            * T: wavelet scales
            * F: graph signal
        Output:
            * C: Hammond's wavelet transform. (len(T) + 1) x len(F)
                 matrix of transform coefficients
    """
    C = []

    for i in range(len(T)):
        C.append([])
        # Each wavelet is represented by an N x N matrix
        for j in range(len(F)):
            dotp = np.dot(F, w[i][j])
            C[i].append(dotp)

    C.append([])
    # Append output of scaling function application at the end
    for j in range(len(F)):
        dotp = np.dot(F, s[j])
        C[-1].append(dotp)

    return np.array(C)


def hammond_wavelets_inverse(w, s, C):
    r"""
        Hammond's wavelet inverse.
        Input:
            * w: wavelets
            * s: low-pass wavelet (scaling function)
            * C: Hammond's wavelet transform. (len(T) + 1) x len(F)
                 matrix of transform coefficients
        Output:
            * F: Reconstructed signal in the vertex domain
    """
    w = np.array(w)
    Wc = np.append(w, np.array([s]), axis=0)
    # Creates copies of the inputs
    nWc = Wc[0, :, :]
    nC = C[0]
    for i in range(1, Wc.shape[0]):
        nWc = np.append(nWc, Wc[i, :, :], axis=0)
        nC = np.append(nC, C[i], axis=0)

    nWc = np.array(nWc)
    nC = np.array(nC)
    # Search a least square solution F, solving:
    # nWc F = nC
    F = np.linalg.lstsq(nWc, nC)[0]

    return F


class Node(object):
    """
        Generic tree-structure used for hierarchical transforms.
    """

    def __init__(self, data):
        """
            Input:
                * data: Anything to be stored in a node.
                        Usually only leaf nodes have data != None
                        data != None often used as stopping condition
        """
        self.data = data
        self.children = []
        self.avgs = []
        self.counts = []
        self.diffs = []
        # Level on the tree. The root has scale = 0
        self.scale = 0
        self.cut = 0
        # count: number of leaves (data != None) in the subtree
        if data is None:
            self.count = 0
        else:
            self.count = 1

    def add_child(self, obj):
        """
            Adds obj as a child to a node.
            Input:
                * obj: anything
        """
        obj.scale = self.scale + 1
        self.children.append(obj)
        self.count = self.count + obj.count


def set_counts(tree):
    """
        Sets counts for intermediate nodes in the tree.
        Input:
            * tree: tree node
        Output:
            * count: count for the tree node
    """
    if tree.data is not None:
        tree.count = 1
        return 1
    else:
        count = 0
        for c in tree.children:
            count = count + set_counts(c)

        tree.count = count

        return count


def laplacian_complete(n):
    """
            Laplacian of a complete graph with n vertices.
            Input:
                    * n: size
            Output:
                    * C: Laplacian
    """
    C = np.ones((n, n))
    C = -1 * C
    D = np.diag(np.ones(n))
    C = (n) * D + C

    return C


def sqrtmi(mat):
    """
        Computes the square-root inverse of a matrix.
        Input:
            * mat: matrix
        Output:
            * square root inverse
    """
    eigvals, eigvecs = eigh(mat)
    eigvecs = eigvecs[:, eigvals > 0]
    eigvals = eigvals[eigvals > 0]

    return dot(eigvecs, dot(diag(1. / sqrt(eigvals)), eigvecs.T))


def sweep(x, G):
    """
        Sweep algorithm for ratio-cut (2nd eigenvector of the Laplacian).
        Based on vector x.
        Input:
            * x: vector
            * G: graph
        Output:
            * vec: indicator vector
    """
    best_val = nx.number_of_nodes(G) - 1
    sorted_x = np.argsort(x)
    size_one = 0
    edges_cut = 0
    nodes_one = {}

    for i in range(x.shape[0]):
        size_one = size_one + 1

        nodes_one[G.nodes()[sorted_x[i]]] = True

        for v in G.neighbors(G.nodes()[sorted_x[i]]):
            if v not in nodes_one:
                edges_cut = edges_cut + 1
            else:
                edges_cut = edges_cut - 1

        den = size_one * (nx.number_of_nodes(G) - size_one)

        if den > 0:
            val = float(edges_cut) / den
            if val <= best_val:
                best_cand = i
                best_val = val

    vec = np.zeros(nx.number_of_nodes(G))

    for i in range(x.shape[0]):
        if i <= best_cand:
            vec[sorted_x[i]] = -1.
        else:
            vec[sorted_x[i]] = 1.

    return vec


def separate_lcc(G, G0):
    """
        Separate vertices in G0 (LCC) from the rest in G returning
        an indicator vector.
        Input:
            * G: Graph
            * G0: Subgraph
        Output:
            * x: indicator vector
    """
    x = []

    for v in G.nodes():
        if v in G0:
            x.append(-1)
        else:
            x.append(1.)

    return np.array(x)


def ratio_cut(G):
    """
        Computes ratio-cut of G based on second eigenvector of the Laplacian.
        Input:
            * G: Graph
        Output:
            * x: Indicator vector
    """

    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    G0 = Gcc[0]

    if nx.number_of_nodes(G) == nx.number_of_nodes(G0):
        x = nx.fiedler_vector(G, method='lobpcg', tol=1e-5)

        x = sweep(x, G)
    else:
        # In case G is not connected
        x = separate_lcc(G, G0)

    return np.array(x)


def get_subgraphs(G, cut):
    """
        Compute subgraphs generated by a cut.
        Input:
            * G: Original graph
            * cut: cut indicator vector
        Output:
            * G1: subgraph 1
            * G2: subgraph 2
    """
    G1 = nx.Graph()
    G2 = nx.Graph()

    i = 0
    P1 = []
    P2 = []
    for v in G.nodes():
        if cut[i] < 0:
            P1.append(v)
        else:
            P2.append(v)
        i = i + 1

    G1 = G.subgraph(P1)
    G2 = G.subgraph(P2)

    return G1, G2


def rc_recursive(node, G, ind):
    """
        Recursively computes ratio-cut.
        Input:
            * node: tree node
            * G: graph
            * ind: vertex index v: unique integer
        Output:
            * none
    """
    if nx.number_of_nodes(G) < 3:
        n = Node(None)
        n.add_child(Node(ind[G.nodes()[0]]))
        n.add_child(Node(ind[G.nodes()[1]]))
        node.add_child(n)
    else:
        C = ratio_cut(G)

        (G1, G2) = get_subgraphs(G, C)

        if nx.number_of_nodes(G1) > 1:
            l = Node(None)
            rc_recursive(l, G1, ind)
            node.add_child(l)
        else:
            l = Node(ind[G1.nodes()[0]])
            node.add_child(l)

        if nx.number_of_nodes(G2) > 1:
            r = Node(None)
            rc_recursive(r, G2, ind)
            node.add_child(r)
        else:
            r = Node(ind[G2.nodes()[0]])
            node.add_child(r)


def ratio_cut_hierarchy(G):
    """
        Computes ratio-cut hierarchy for a graph.
        Input:
            * G: graph
        Output:
            * root: tree root
            * ind: graph index v: unique integer
    """
    i = 0
    ind = {}
    for v in G.nodes():
        ind[v] = i
        i = i + 1

    root = Node(None)

    rc_recursive(root, G, ind)

    return root, ind


def compute_coefficients(tree, F):
    """
        Computes tree coefficients for Gavish's transform.
        Input:
            * tree: tree
            * F: graph signal
        Output:
            * None
    """
    if tree.data is None:
        avg = 0
        count = 0
        for i in range(len(tree.children)):
            compute_coefficients(tree.children[i], F)
            avg = avg + tree.children[i].avg * tree.children[i].count
            count = count + tree.children[i].count

            if i > 0:
                tree.avgs.append(float(avg) / count)
                tree.counts.append(count)
                tree.diffs.append(2 * tree.children[i].count *
                                  (tree.children[i].avg - float(avg) / count))
        tree.avgs = list(reversed(tree.avgs))
        tree.avg = float(avg) / tree.count
    else:
        tree.avg = F[tree.data]


def reconstruct_values(tree, F):
    """
            Reconstructs values for Gavish's transform based on a tree.
            Input:
                    * tree: tree
                    * F: graph signal
            Output:
                    * None
    """
    if tree.data is None:
        avg = tree.avg * tree.count
        count = tree.count
        for i in reversed(range(len(tree.children))):
            if i == 0:
                tree.children[i].avg = avg / tree.children[i].count
                reconstruct_values(tree.children[i], F)
            else:
                tree.children[i].avg = float(avg) / count + 0.5 * \
                    float(tree.diffs[i - 1]) / tree.children[i].count
                reconstruct_values(tree.children[i], F)
                count = count - tree.children[i].count
                avg = avg - tree.children[i].avg * tree.children[i].count
                tree.avgs.append(float(avg) / count)

        tree.avgs = list(reversed(tree.avgs))
    else:
        F[tree.data] = tree.avg


def clear_tree(tree):
    """
        Clears tree info.
        Input:
            * tree
        Output:
            * None
    """
    tree.avg = 0
    tree.diffs = []
    tree.avgs = []

    if tree.data is None:
        for i in range(len(tree.children)):
            clear_tree(tree.children[i])


def get_coefficients(tree, wtr):
    """
        Recovers wavelet coefficients from the wavelet tree.
        Input:
            * tree
            * wtr: wavelet coefficients
        Output:
            * None
    """
    Q = deque()
    scales = []
    wtr.append(tree.count * tree.avg)

    Q.append(tree)

    while len(Q) > 0:
        node = Q.popleft()
        scales.append(node.scale)

        for j in range(len(node.diffs)):
            wtr.append(node.diffs[j])

        for i in range(len(node.children)):
            Q.append(node.children[i])


def set_coefficients(tree, wtr):
    """
        Sets wavelet tree coefficients.
        Input:
            * tree
            * wtr: wavelet coefficients
    """
    Q = deque()
    tree.avg = float(wtr[0]) / tree.count
    p = 1
    Q.append(tree)

    while len(Q) > 0:
        node = Q.popleft()

        for j in range(len(node.children) - 1):
            node.diffs.append(wtr[p])
            p = p + 1

        for i in range(len(node.children)):
            Q.append(node.children[i])


def gavish_wavelet_transform(tree, ind, G, F):
    """
        Gavish's wavelet transform.
        Input:
            * tree
            * ind: vertex index v : unique integer
            * G: graph
            * F: graph signal
        Output:
            * wtr: wavelet transform.
    """
    wtr = []
    clear_tree(tree)
    compute_coefficients(tree, F)
    get_coefficients(tree, wtr)

    return np.array(wtr)


def gavish_wavelet_inverse(tree, ind, G, wtr):
    """
        Gavish's wavelet inverse.
        Input:
            * tree
            * ind: vertex index v: unique integer
            * G: graph
            * wtr: wavelet transform
        Output:
            * F: wavelet inverse
    """
    F = []

    for i in range(len(G.nodes())):
        F.append(0)

    clear_tree(tree)
    set_coefficients(tree, wtr)
    reconstruct_values(tree, F)

    return np.array(F)
