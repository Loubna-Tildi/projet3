import numpy as np
import matplotlib.pyplot as plt
from utils.distances import euclidean_distance, manhattan_distance, cosine_distance


class KmeansClustering:
    def __init__(self, k=3, distance="Euclidean"):
        self.k = k
        self.distance = distance
        self.centroids = None

    def initialize_centroids(self, Data):
        """Initialize centroids as random data points from the dataset."""
        self.centroids = np.random.uniform(
            np.amin(Data, axis=0), np.amax(Data, axis=0), size=(self.k, Data.shape[1])
        )

    def assign_clusters(self, Data):
        """Assign data points to the nearest centroid."""
        clusters = []
        for data_point in Data:
            if self.distance == "Euclidean":
                distances = euclidean_distance(data_point, self.centroids)
            elif self.distance == "Manhattan":
                distances = manhattan_distance(data_point, self.centroids)
            elif self.distance == "Cosine":
                distances = cosine_distance(data_point, self.centroids)
            cluster_num = np.argmin(distances)
            clusters.append(cluster_num)
        return np.array(clusters)

    def update_centroids(self, Data, labels):
        """Update centroids to the mean of assigned data points."""
        new_centroids = np.array([Data[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, Data, max_iter=100):
        """Fit the K-means model to the data."""
        self.initialize_centroids(Data)
        for _ in range(max_iter):
            labels = self.assign_clusters(Data)
            new_centroids = self.update_centroids(Data, labels)
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < 0.0001):
                break
            else:
                self.centroids = new_centroids
        return labels

    def plot_clusters(self, Data, labels):
        """Visualize the clusters and centroids."""
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                  "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
        fig, ax = plt.subplots()
        for i in range(self.k):
            points = Data[labels == i]
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i % len(colors)], label=f'Cluster {i + 1}')
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='*', s=120, c='#000000', label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
        return fig
