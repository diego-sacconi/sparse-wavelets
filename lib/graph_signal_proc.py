import math
import random
import sys
from collections import deque

import networkx as nx
import numpy as np
from numpy import dot, diag, sqrt
from numpy.linalg import eigh
import scipy.optimize
from scipy import linalg
import scipy.fftpack

from sklearn.preprocessing import normalize


def compute_eigenvectors_and_eigenvalues(L):
    """
            Computes eigenvectors and eigenvalues of the matrix L
            Input:
                    * L: matrix
            Output:
                    * U: eigenvector matrix, one vector/column, sorted by corresponsing eigenvalue
                    * lamb: eigenvalues, sorted in increasing order
    """
    lamb, U = linalg.eig(L)

    idx = lamb.argsort()
    lamb = lamb[idx]
    U = U[:, idx]

    return U, lamb


def s(x):
    """
            Cubic spline.
            Input:
                    * x
            Output:
                    * spline(x)
    """
    return -5 + 11 * x - 6 * pow(x, 2) + pow(x, 3)


def g(x):
    """
            Wavelet kernel.
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
        return s(x)
    else:
        return pow(x_2, b) * pow(x, -b)


def comp_gamma():
    """
            Computes gamma function
            Input:
                    * None
            Output:
                    * Gamma function (array)
    """
    def gn(x): return -1 * g(x)
    xopt = scipy.optimize.fminbound(gn, 1, 2)
    return xopt


def h(x, gamma, lamb_max, K):
    """
            Scaling function (see details in the paper "Graph wavelets via spectral theory".
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
    """
            Computes wavelet scales
            Input:
                    * lamb_max: upper bound spectrum
                    * K: normalization
                    * J: number of scales
            Output:
                    * scales array
    """
    lamb_min = float(lamb_max) / K
    s_min = float(1) / lamb_max
    s_max = float(2) / lamb_min

    return np.exp(np.linspace(math.log(s_max), math.log(s_min), J))


def graph_low_pass(lamb, U, N, T, gamma, lamb_max, K):
    """
            Low-pass filter.
            Input:
                    * lamb: eigenvalues
                    * U: eigenvector matrix
                    * N: number of nodes
                    * T: wavelet scales
                    * gamma:
                    * lamb_max: upper-bound spectrum
                    * K: normalization
            Output:
                    * s: Low-pass filter as a #vertices x #vertices matrix
    """
    s = []

    for n in range(0, len(N)):
        s.append([])

    for n in range(0, len(N)):
        for m in range(0, len(U)):
            s_n_m = 0

            for x in range(0, len(U)):
                s_n_m = s_n_m + U[n][x] * U[m][x] * h(T[-1] * lamb[x], gamma, lamb_max, K)

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
                    * w: wavelets as a #vertices x #vertices x #scales matrix
    """
    w = []

    for t in range(0, len(T)):
        w.append([])
        for n in range(0, len(N)):
            w[t].append([])

    for t in range(0, len(T)):
        for n in range(0, len(N)):
            for m in range(0, len(U)):
                w_t_n_m = 0

                for x in range(0, len(U)):
                    w_t_n_m = w_t_n_m + U[n][x] * U[m][x] * g(T[t] * lamb[x])

                w[t][n].append(w_t_n_m)

    return w


def graph_fourier(F, U):
    """
            Graph Fourier transform.
            Input:
                    * F: Graph signal as a #vertices size array, values ordered by G.nodes()
                    * U: Eigenvectors matrix
            Ouput:
                    * lambdas: Graph Fourier transform
    """
    lambdas = []

    for i in range(0, len(U)):
        lambdas.append(np.dot(F, U[:, i]))

    lambdas = np.array(lambdas)

    return lambdas


def graph_fourier_inverse(GF, U):
    """
            Graph Fourier inverse:
            Input:
                    * GF: Graph fourier transform
                    * U: Eigenvectors matrix
            Output:
                    * F: Inverse
    """
    F = np.zeros(U.shape[0])
    for v in range(U.shape[0]):
        for u in range(U.shape[1]):
            F[v] = F[v] + (GF[u] * U[v][u]).real

    return F


def hammond_wavelet_transform(w, s, T, F):
    """
            Hammond wavelet transform.
            Input:
                    * w: wavelets
                    * s: low-pass wavelets
                    * T: wavelet scales
                    * F: graph signal
            Output:
                    * C: Hammond's wavelet transform
    """
    C = []

    for i in range(len(T)):
        C.append([])
        for j in range(len(F)):
            dotp = np.dot(F, w[i][j])
            C[i].append(dotp)

    C.append([])
    for j in range(len(F)):
        dotp = np.dot(F, s[j])
        C[-1].append(dotp)

    return np.array(C)


def hammond_wavelets_inverse(w, s, C):
    """
            Hammond's wavelet inverse.
            Input:
                    * w: wavelets
                    * s: low-pass wavelets
                    * C: transform
            Output:
                    * F: inverse
    """
    w = np.array(w)
    Wc = np.append(w, np.array([s]), axis=0)

    nWc = Wc[0, :, :]
    nC = C[0]
    for i in range(1, Wc.shape[0]):
        nWc = np.append(nWc, Wc[i, :, :], axis=0)
        nC = np.append(nC, C[i], axis=0)

    nWc = np.array(nWc)
    nC = np.array(nC)

    F = np.linalg.lstsq(nWc, nC)[0]

    return F


class Node(object):
    """
            Generic tree-structure used for hierarchical transforms.
    """

    def __init__(self, data):
        """
                Initialization.
                Input:
                        * data: Anything to be stored in a node
        """
        self.data = data
        self.children = []
        self.avgs = []
        self.counts = []
        self.diffs = []
        self.scale = 0
        self.ftr = []
        self.L = []
        self.U = []
        self.cut = 0

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


def get_children(tree, part, G):
    """
            Recursively gets all the children of a given node.
            Input:
                    * tree: tree node
                    * part: list that will contain children
                    * G: graph
            Output:
                    * None
    """
    if tree.data is not None:
        part.append(G.nodes()[tree.data])
    else:
        for c in tree.children:
            get_children(c, part, G)


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


def partitions_level_rec(tree, level, G, l, partitions):
    """
            Recursively extracts partitions up to a certain level in the tree.
            Input:
                    * tree: tree
                    * level: max level
                    * G: graph
                    * l: current level
                    * partitions: partitions recovered
            Output:
                    None
    """
    if l >= level:
        part = []
        get_children(tree, part, G)
        if len(part) > 0:
            partitions.append(part)
    else:
        if tree.data is None:
            for c in tree.children:
                partitions_level_rec(c, level, G, l + 1, partitions)
        else:
            partitions.append([tree.data])


def partitions_level(tree, level, G):
    """
            Recovers partitions at a certain level of the three.
            Input:
                    * tree: tree
                    * level: level
                    * G: graph
            Output:
                    * partitions: set of vertices in each partition
    """
    partitions = []
    partitions_level_rec(tree, level, G, 0, partitions)

    return partitions


def build_matrix(G, ind):
    """
            Builds graph distance matrix.
            Input:
                    * G: graph
                    * ind: dictionary vertex: unique integer
            Output:
                    * M: matrix
    """
    M = []
    dists = nx.all_pairs_dijkstra_path_length(G)

    M = np.zeros((len(G.nodes()), len(G.nodes())))

    for v1 in G.nodes():
        for v2 in G.nodes():
            M[ind[v1]][ind[v2]] = dists[v1][v2]

    return M


def select_centroids(M, radius):
    """
            Selects half of the vertices as centroids.
            Input:
                    * M
                    * radius
            Output:
                    * centroids
    """
    nodes = list(range(M.shape[0]))
    random.shuffle(nodes)
    nodes = nodes[:int(len(nodes) / 2)]
    cents = [nodes[0]]
    mn = sys.float_info.min

    for i in range(1, len(nodes)):
        add = True
        for j in range(len(cents)):
            if M[cents[j]][nodes[i]] <= radius * mn:
                add = False
                break
        if add:
            cents.append(nodes[i])

    return cents


def coarse_matrix(M, H, cents, nodes):
    """
            Makes matrix coarser based on centroids.
            Input:
                    * M: distance matrix
                    * H:
                    * cents: centroids
                    * nodes: list of nodes
            Output:
                    * Q: new matrix
                    * J
                    * new_nodes: new node list
    """
    Q = np.zeros((len(cents), len(cents)))
    J = []
    assigns = []
    new_nodes = []

    for i in range(len(cents)):
        J.append([])
        assigns.append([])
    new_nodes.append(Node(None))

    for i in range(M.shape[0]):
        min_dist = M[i][cents[0]]
        min_cent = 0

        for j in range(1, len(cents)):
            if M[i][cents[j]] < min_dist:
                min_dist = M[i][cents[j]]
                min_cent = j

        J[min_cent].append(H[i])
        assigns[min_cent].append(i)
        new_nodes[min_cent].add_child(nodes[i])

    for i in range(len(cents)):
        if len(new_nodes[i].children) == 1:
            new_nodes[i] = new_nodes[i].children[0]

        for j in range(len(cents)):
            if i != j:
                for m in assigns[i]:
                    for k in assigns[j]:
                        Q[i][j] = Q[i][j] + pow(M[m][k], 2)

    Q = normalize(Q, axis=1, norm='l1')

    return Q, J, new_nodes


def get_partitions(x, node_list):
    """
            Gets partitions given indicator vector.
                    if x < 0: partition 1
                    if x <= 0: partition 2
            Input:
                    * node_list: list of nodes
                    * x: indicator vector
            Output:
                    * P1: partition 1
                    * P2: partition 2
    """
    P1 = []
    P2 = []

    for i in range(x.shape[0]):
        if x[i] < 0:
            P1.append(node_list[i])
        else:
            P2.append(node_list[i])

    return P1, P2


def get_new_laplacians(L, P1, P2, ind):
    """
            Compute new Laplacian matrices for partitions P1 and P2.
            Input:
                    * L: Higher-level laplacian
                    * P1: partition 1
                    * P2: partition 2
                    * ind: node index vertex: unique integer
            Output:
                    * L1: Laplacian P1
                    * L2: Laplacian P2
    """
    data = []
    row = []
    col = []

    for i in range(len(P1)):
        d = 0
        for j in range(len(P1)):
            if i != j and L[ind[P1[i]], ind[P1[j]]] != 0:
                row.append(i)
                col.append(j)
                data.append(float(L[ind[P1[i]], ind[P1[j]]]))
                d = d - L[ind[P1[i]], ind[P1[j]]]

        row.append(i)
        col.append(i)
        data.append(float(d))

    L1 = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(P1), len(P1)))

    data = []
    row = []
    col = []

    for i in range(len(P2)):
        d = 0
        for j in range(len(P2)):
            if i != j and L[ind[P2[i]], ind[P2[j]]] != 0:
                row.append(i)
                col.append(j)
                data.append(float(L[ind[P2[i]], ind[P2[j]]]))
                d = d - L[ind[P2[i]], ind[P2[j]]]

        row.append(i)
        col.append(i)
        data.append(float(d))

    L2 = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(P2), len(P2)))

    return L1, L2


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


def sqrtm(mat):
    """
            Matrix square root.
            Input:
                    * mat: matrix
            Output:
                    * matrix square root
    """
    eigvals, eigvecs = eigh(mat)

    eigvecs = eigvecs[:, eigvals > 0]
    eigvals = eigvals[eigvals > 0]

    return dot(eigvecs, dot(diag(sqrt(eigvals)), eigvecs.T))


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


def create_linked_list(L):
    """
            Creates linked list from a Laplacian matrix.
            Input:
                    * L: matrix
            Output:
                    * linked_list: linked list
    """
    linked_list = {}

    for i in L.nonzero()[0]:
        linked_list[i] = []
        for j in range(L.shape[1]):
            if L[i, j] < 0:
                linked_list[i].append(j)
    return linked_list


def sweep(x, G):
    """
            Sweep algorithm for ratio-cut (2nd eigenvector of the Laplacian) based on vector x.
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
        else:
            val = nx.number_of_nodes(G)

        if val <= best_val:
            best_cand = i
            best_val = val

    vec = []

    vec = np.zeros(nx.number_of_nodes(G))

    for i in range(x.shape[0]):
        if i <= best_cand:
            vec[sorted_x[i]] = -1.
        else:
            vec[sorted_x[i]] = 1.

    return vec


def separate_lcc(G, G0):
    """
            Separates vertices in G0 (LCC) from the rest in G using indicator vector.
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


def eig_vis_rc(G):
    """
            Second and third eigenvectors of the graph Laplacian. For visualization.
            Input:
                    * G: Graph
            Output:
                    * x1: Second eigenvector
                    * x2: Third eigenvector
    """
    L = nx.laplacian_matrix(G).todense()
    (eigvals, eigvecs) = scipy.linalg.eigh(L, eigvals=(1, 2))

    x1 = np.asarray(eigvecs[:, 0])
    x2 = np.asarray(eigvecs[:, 1])

    return x1, x2


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


def gavish_hierarchy(G, radius):
    """
            Builds Gavish's hierarchy of a graph.
            Input:
                    * G: graph
                    * radius: radius
            Output:
                    * tree root
                    * ind: vertex index v: unique integer
    """
    H = []
    nodes = []
    ind = {}
    i = 0
    for v in G.nodes():
        ind[v] = i
        nodes.append(Node(i))
        H.append(i)
        i = i + 1

    M = build_matrix(G, ind)

    while M.shape[0] > 1:
        cents = select_centroids(M, radius)
        Q, J, new_nodes = coarse_matrix(M, H, cents, nodes)
        M = Q
        H = J
        nodes = new_nodes

    return nodes[0], ind


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


def get_cut_sizes(tree):
    """
            Recovers cut sizes from tree.
            Input:
                    * tree
            Output:
                    * None
    """
    Q = deque()
    cut_sizes = []

    Q.append(tree)

    while len(Q) > 0:
        node = Q.popleft()
        cut_sizes.append(node.cut)

        for i in range(len(node.children)):
            Q.append(node.children[i])

    return cut_sizes


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
