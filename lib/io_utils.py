"""
This module read graph's info from files with the following format:

input_graph_name has info about edges.
Row format: "vertex_A, vertex_B[, edge_weight]"

input_data_name has info about the graph signal.
Row format : "vertex_id, vertex_value"
"""

import networkx as nx
import numpy as np


def read_graph(input_graph_name, input_data_name):
    """
        Read graph from file.
        Input:
            * input_graph_name has info about edges.
                Row format: "vertex_A, vertex_B[, edge_weight]"
            * input_data_name has info about the graph signal.
                Row format : "vertex_id, vertex_value"
        Output:
            * networkx graph
    """

    # Reading input data
    input_data = open(input_data_name, 'r')
    values = {}

    for line in input_data:
        line = line.rstrip()
        vec = line.rsplit(',')

        vertex = vec[0]
        value = float(vec[1])
        values[vertex] = value

    input_data.close()

    # Reading graph data
    G = nx.Graph()
    input_graph = open(input_graph_name, 'r')

    for line in input_graph:
        line = line.rstrip()
        vec = line.rsplit(',')
        v1 = vec[0]
        v2 = vec[1]
        # Note that the edge weight is always set to 1
        # even when provided available in input_graph_name
        if v1 in values and v2 in values:
            G.add_edge(v1, v2, weight=1.)

    # Extracting largest connected component from graph
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)

    G = Gcc[0]

    graph_signal = {}

    # Setting the graph_signal as node attribute
    for v in values.keys():
        if v in G:
            graph_signal[v] = values[v]

    input_graph.close()
    nx.set_node_attributes(G, "value", graph_signal)

    return G


def read_values(input_data_name, G):
    """
        Read the graph signal from file
        Input:
            * input_data_name has info about the graph signal.
                Row format : "vertex_id, vertex_value"
            * G: networkx graph
        Output:
            * F: normalized node values array, ordered by G.nodes()
    """
    graph_signal = {}
    input_data = open(input_data_name, 'r')

    # Reading file
    for line in input_data:
        line = line.rstrip()
        vec = line.rsplit(',')

        vertex = vec[0]
        value = float(vec[1])
        graph_signal[vertex] = value

    input_data.close()

    F = []
    for v in G.nodes():
        if v in graph_signal:
            F.append(float(graph_signal[v]))
        else:
            F.append(0.)

    # Normalization
    F = np.array(F)
    F = F / np.max(F)
    F = F - np.mean(F)

    return F
