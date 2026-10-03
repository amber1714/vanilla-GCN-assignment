import networkx as nx
import torch


def load_karate_club():
    graph = nx.karate_club_graph()
    n = graph.number_of_nodes()
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    for u, v in graph.edges():
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0
    features = torch.eye(n, dtype=torch.float32)
    labels = torch.tensor([0 if graph.nodes[node]["club"] == "Mr. Hi" else 1 for node in range(n)], dtype=torch.long)
    train_idx = torch.tensor([0, 1, 2, 33], dtype=torch.long)
    val_idx = torch.tensor([3, 8, 30, 31], dtype=torch.long)
    all_idx = torch.arange(n)
    mask = torch.ones(n, dtype=torch.bool)
    mask[train_idx] = False
    mask[val_idx] = False
    test_idx = all_idx[mask]
    return graph, adjacency, features, labels, train_idx, val_idx, test_idx
