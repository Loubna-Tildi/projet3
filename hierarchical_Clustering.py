from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


class hierarchicalClustering:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    def fit_predict(self, Data):
        self.labels = self.model.fit_predict(Data)
        self.linked = linkage(Data, method=self.linkage)
        return self.labels

    def plot_dendrogram(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        dendrogram(self.linked, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Distance')
        return fig

    def plot_clusters(self, Data, labels):
        fig, ax = plt.subplots()
        scatter = ax.scatter(Data[:, 0], Data[:, 1], c=labels, cmap='viridis', label='Data Points')
        ax.set_title('Hierarchical Clustering Results')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        return fig
