import numpy as np, h5py
import matplotlib.pyplot as plt
import networkx as nx

def graph_from_adjancency_matrix(adjancency_matrix):
    g = nx.from_numpy_matrix(adjancency_matrix)
    return g

def draw_graph(g):
    if not type(g) == nx.classes.graph.Graph:
        print('Attempting to draw none graph type object')
        return False
    nx.draw_spring(g, node_size=50)
    plt.show()
