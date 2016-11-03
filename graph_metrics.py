import numpy as np

def closed_walks(adjancency_matrix, max_walk_length=10):
    """
    Calculates [x(2),...,x(kmax)] where x(k) is the number of walks of length k
    from each node to itself. This encompasses all closed walks of length k,
    including backtracks.
    i.e. (u -> v -> u -> v -> u) as a closed walk length k = 4
    Calculated using Tr(A^k) where A is the adjancency matrix.
    ith entry of feature vector = Tr(A^i)
    ----------------------------------------------------------------------------
    adjancency_matrix: a square matrix
    max_walk_length: the max walk length to consider
    ----------------------------------------------------------------------------
    return
        feature_vector: np.ndarray, length k-1 vector of number of closed walks
    """

    if not max_walk_length:
        prompt = 'Enter the max walk length to consider for all closed walks : '
        max_walk_length = int(input(prompt))

    # keys the power of adjancency_matrix
    a_powers = {
        1: adjancency_matrix
    }

    for i in range(2, max_walk_length + 1):
        a_powers[i] = np.dot(a_powers[i - 1], adjancency_matrix)

    return np.array(
        [np.trace(a_powers[i]) for i in range(2, max_walk_length + 1)]
    )

def olg(adjancency_matrix):
    """
    The perron-frobenius operator is the adjacency matrix of the OLG  of some
    graph G. The adjacency matrix of OLG (T) is constructed by considering each
    edge in the original G described by adjacency matrix A.
    OLG - oriented line graph
    ----------------------------------------------------------------------------
    adjancency_matrix: a square adjancency matrix of the graph for which
    perron-frobenius operator is required
    ----------------------------------------------------------------------------
    t: the oriented line graph adjacency matrix/perron-frobenius operator
    """
    a = adjancency_matrix
    n = a.shape[0] # dimension of adjancency matrix

    # v the vertex list for olg, each vertex is an edge in a
    v = [(i,j)  for i in range(n) for j in range(n) if a[i][j] == 1]

    # perron frobenius has dimension |e|x|e|
    t = np.zeros((len(v),len(v)))

    # build adjancency matrix for olg by considering each edge in vertex list
    for e in v:

        # For this particular edge e (vertex for T) build list of edges
        # (vertices for T) that start where this e ends, disallowing any cycles
        # i.e. (1,2) -> (2,3) true, (0,1) -> (1,0) false
        olg_condition = lambda e1, e2 : e1[1] == e2[0] and e2[1] != e1[0]

        # store edges (vertices for T) that satisfy the olg condition
        adjacent_edges = [e2 for e2 in v if olg_condition(e, e2)]

        # adjacent edges (T vertices) satisfying the olg condition should be
        # adjacent to the current edge (T vertex) in T
        for e2 in olg_edges:
            t[v.index(e)][v.index(e2)] = 1

    return t
