import torch


def add_self_loops(adjacency: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(adjacency.size(0), dtype=adjacency.dtype, device=adjacency.device)
    return adjacency + identity


def degree_matrix(adjacency_with_loops: torch.Tensor) -> torch.Tensor:
    degrees = adjacency_with_loops.sum(dim=1)
    return torch.diag(degrees)


def inverse_sqrt_degree(degree: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diagonal = torch.diag(degree).clamp_min(eps)
    return torch.diag(torch.pow(diagonal, -0.5))


def symmetric_normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    a_hat = add_self_loops(adjacency)
    d_hat = degree_matrix(a_hat)
    d_inv_sqrt = inverse_sqrt_degree(d_hat)
    return d_inv_sqrt @ a_hat @ d_inv_sqrt


def linear_feature_transform(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return features @ weights


def neighborhood_aggregation(normalized_adjacency: torch.Tensor, transformed_features: torch.Tensor) -> torch.Tensor:
    return normalized_adjacency @ transformed_features


def gcn_layer_formula(normalized_adjacency: torch.Tensor, features: torch.Tensor, weights: torch.Tensor, activation=None) -> torch.Tensor:
    output = normalized_adjacency @ features @ weights
    return activation(output) if activation is not None else output


def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels)
