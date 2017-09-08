import networkx as nx
import numpy as np


def read_graph(input_graph_name, input_data_name):
    """
        Reads graph from file.
        Input:
            * input_graph_name: csv edge list
            * input_data_name: csv node-value pairs
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

        if v1 in values and v2 in values:
            G.add_edge(v1, v2, weight=1.)

    # Extracting largest connected component from graph
    Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)

    G = Gcc[0]

    values_in_graph = {}

    # Setting values as node attributes
    for v in values.keys():
        if v in G:
            values_in_graph[v] = values[v]

    input_graph.close()
    nx.set_node_attributes(G, "value", values_in_graph)

    return G


def read_values(input_data_name, G):
    """
        Reads node values.
        Input:
            * input_data_name: csv node-value pairs
            * G: networkx graph
        Output:
            * F: normalized node values array, ordered by G.nodes()
    """
    D = {}
    input_data = open(input_data_name, 'r')

    # Reading file
    for line in input_data:
        line = line.rstrip()
        vec = line.rsplit(',')

        vertex = vec[0]
        value = float(vec[1])
        D[vertex] = value

    input_data.close()

    F = []
    for v in G.nodes():
        if v in D:
            F.append(float(D[v]))
        else:
            F.append(0.)

    # Normalization
    F = np.array(F)
    F = F / np.max(F)
    F = F - np.mean(F)

    return F
