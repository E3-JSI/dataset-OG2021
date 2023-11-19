import math
import torch

from typing import List, Tuple

# ===============================================
# Distance Methods
# ===============================================


def cosine_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> float:
    """Calculates the cosine similarity between two vectors
    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.
    Returns:
        cosine_similarity (float): The cosine similarity between the vectors.
    """
    return vector1.dot(vector2).item()


def avg_cosine_similarity(
    vectors: List[torch.Tensor], vector_prime: torch.Tensor
) -> float:
    cosines = [cosine_similarity(vector, vector_prime) for vector in vectors]
    return sum(cosines) / len(cosines)


def jaccard_index(s1: set, s2: set) -> float:
    """Gets the Jaccard Index
    Calculates the Jaccard Index using the following equation:
        Jaccard(s1, s2) = \\frac{s1 \\cap s2}{s1 \\cup s2}
    Args:
        s1 (set): The first set.
        s2 (set): The second set.
    Returns:
        jaccard_index (float): The Jaccard Index of the two sets.
    """
    return len(s1 & s2) / len(s1 | s2)


def get_intra_distances(embeds: List[torch.Tensor], centroid: torch.Tensor) -> dict:
    if len(embeds) < 2:
        return {"maximum": 1.0, "average": 1.0, "centroid": 1.0, "distances": []}

    X = torch.cat(tuple([embed.unsqueeze(0) for embed in embeds]), 0)
    S = 1 - torch.matmul(X, X.T)

    # intra-cluster distance
    maximum = torch.max(S)
    average = torch.sum(S) / (S.shape[0] * (S.shape[1] - 1))

    # centroid diameter distance
    dists = 1 - torch.matmul(X, centroid)
    c_dist = torch.sum(dists) / X.shape[0]

    return {
        "maximum": maximum.item(),
        "average": average.item(),
        "centroid": c_dist.item(),
        "distances": dists.tolist(),
    }


# ===============================================
# Cluster Methods
# ===============================================


def get_centroid(embeds: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    """Calculates the centroid
    Calculates the centroid with the following equations:
        c = \\frac{sum_{i=1}^{k} a_{i}}{k}
        c = \\frac{c}{||c||}
    Args:
        embeds (List[torch.Tensor]): The embedding list of articles
            in the cluster.
    Returns:
        centroid (torch.Tensor): The normalized centroid.
        c_norm (float): The centroids norm before normalization.
    """
    X = torch.cat(tuple([embed.unsqueeze(0) for embed in embeds]), 0)
    centroid = torch.sum(X, 0) / X.shape[0]
    # calculate the centroid norm
    c_norm = torch.linalg.vector_norm(centroid, ord=2).item()
    # normalize the centroid
    centroid = centroid / c_norm
    return centroid, c_norm


def update_centroid(
    centroid: torch.Tensor, c_norm: float, n_articles: int, a_embed: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Updates the centroid
    Updates the centroid with the following equations:
        c_i = \\frac{n_{i-1} * ||c_{i-1}|| * c_{i-1} + a_{i}}{n_{i}}
        c_i = \\frac{c_i}{||c_i||}
    Args:
        centroid (torch.Tensor): The current centroids tensor. Corresponds to
            c_{i-1} in the equation.
        c_norm (float): The current centroids norm. Corresponds to ||c_{i-1}||
            in the equation.
        n_articles (int): The previous number of articles in the cluster.
            Corresponds to n_{i-1} in the equation.
        a_embed (torch.Tensor): The new articles tensor. Corresponds to a_{i}
            in the equation.
    Returns:
        centroid (torch.Tensor): The updated normalized centroid.
        c_norm (float): The updated centroids norm before normalization.
    """
    centroid *= n_articles * c_norm
    centroid += a_embed
    centroid /= n_articles + 1
    c_norm = torch.linalg.vector_norm(centroid, ord=2).item()
    centroid = centroid / c_norm
    return centroid, c_norm


# ===============================================
# Statistics Methods
# ===============================================


def get_max(vals: List[float]) -> float:
    return max(vals)


def get_min(vals: List[float]) -> float:
    return min(vals)


def get_avg(vals: List[float]) -> float:
    return sum(vals) / len(vals)


def get_var(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0
    avg = get_avg(vals)
    return sum([math.pow(v - avg, 2) for v in vals]) / (len(vals) - 1)


def get_std(vals: List[float]) -> float:
    return math.sqrt(get_var(vals))
