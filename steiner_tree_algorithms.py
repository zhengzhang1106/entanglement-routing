import networkx as nx


def approximate_steiner_tree(graph, terminals):
    """
    Approximate Steiner Tree using metric closure + MST on complete graph.
    Returns a subgraph (nx.Graph) of the original topology.
    """
    # Step 1: Metric closure - complete graph with shortest path lengths
    metric_closure = nx.Graph()
    for u in terminals:
        for v in terminals:
            if u == v:
                continue
            try:
                length = nx.shortest_path_length(graph, u, v, weight='length')
                metric_closure.add_edge(u, v, weight=length)
            except nx.NetworkXNoPath:
                continue

    # Step 2: MST of the complete terminal graph
    mst = nx.minimum_spanning_tree(metric_closure, weight='weight')

    # Step 3: Map MST edges back to paths in original graph
    steiner_tree = nx.Graph()
    for u, v in mst.edges():
        try:
            path = nx.shortest_path(graph, source=u, target=v, weight='length')
            for i in range(len(path) - 1):
                steiner_tree.add_edge(path[i], path[i + 1])
        except nx.NetworkXNoPath:
            continue

    return steiner_tree


def has_connecting_tree(subgraph, user_set):
    """
    Checks if a connecting tree exists among the users in the given subgraph.
    This is a proxy for whether a routing solution can be found.
    """
    if len(user_set) <= 1:
        return True

    # Check if all users are present in the subgraph before proceeding
    if not all(node in subgraph.nodes for node in user_set):
        return False

    # Check if all users are in the same connected component
    try:
        subgraph_of_users = subgraph.subgraph(user_set)
        return nx.is_connected(subgraph_of_users)
    except nx.NetworkXError:
        # This error occurs if some users are not in the subgraph
        return False