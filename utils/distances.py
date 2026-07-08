import numpy as np


def euclidean_distance(point, centroids):
    """Distance euclidienne entre un point et un ensemble de centroïdes."""
    return np.sqrt(np.sum((centroids - point) ** 2, axis=1))


def manhattan_distance(point, centroids):
    """Distance de Manhattan entre un point et un ensemble de centroïdes."""
    return np.sum(np.abs(centroids - point), axis=1)


def cosine_distance(point, centroids):
    """Distance cosinus (1 - similarité cosinus) entre un point et des centroïdes."""
    point_norm = point / (np.linalg.norm(point) + 1e-10)
    centroids_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    similarity = np.dot(centroids_norm, point_norm)
    return 1 - similarity
