import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

from kmeans_Clustering import KmeansClustering
from hierarchical_Clustering import hierarchicalClustering
from style import styles

styles()


# Implémentation améliorée d'ElbowMethod
class ElbowMethod:
    def __init__(self, data, k_range=(1, 10)):
        self.data = data
        self.k_min, self.k_max = k_range
        self.inertia_values = []
        self.k_values = list(range(self.k_min, self.k_max + 1))

    def fit(self):
        """Calculate inertia for different k values."""
        self.inertia_values = []
        for k in self.k_values:
            if k <= 1:  # KMeans special case for k=1
                if k == 0:
                    self.inertia_values.append(float('inf'))
                else:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(self.data)
                    self.inertia_values.append(kmeans.inertia_)
                continue
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.data)
            self.inertia_values.append(kmeans.inertia_)
        return self

    def find_optimal_k(self):
        """Find the optimal k using the elbow method."""
        if not self.inertia_values:
            self.fit()
        if len(self.inertia_values) <= 2:
            return 2  # Default if not enough points

        # Simple elbow detection - find the k where the rate of decrease sharply changes
        inertia_diffs = np.diff(self.inertia_values)
        inertia_diffs2 = np.diff(inertia_diffs)
        optimal_idx = np.argmax(inertia_diffs2) + 1  # +1 because of double diff

        # Ensure the index is within valid range
        if optimal_idx >= len(self.k_values):
            optimal_idx = len(self.k_values) - 1
        return self.k_values[optimal_idx]

    def plot_elbow_curve(self):
        """Plot the elbow curve using Plotly."""
        if not self.inertia_values:
            self.fit()

        optimal_k = self.find_optimal_k()
        optimal_idx = self.k_values.index(optimal_k)

        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.k_values,
            y=self.inertia_values,
            mode='lines+markers',
            name='Inertia',
            marker=dict(size=8, color='blue')
        ))

        # Add point for optimal k
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[self.inertia_values[optimal_idx]],
            mode='markers',
            name=f'Optimal k = {optimal_k}',
            marker=dict(size=12, color='red', symbol='star')
        ))

        # Update layout
        fig.update_layout(
            title='Elbow Method for Optimal k',
            xaxis=dict(title='Number of clusters (k)', tickmode='linear'),
            yaxis=dict(title='Inertia'),
            legend=dict(x=0.7, y=0.9),
            hovermode='closest',
            height=600
        )
        return fig


# Functions for Plotly visualizations
def plot_clusters_2d_plotly(data, labels, title="2D Cluster Visualization"):
    """Plot clusters in 2D using Plotly."""
    if data.shape[1] < 2:
        st.error("Need at least 2 dimensions for 2D visualization.")
        return None
    df = pd.DataFrame(data[:, :2], columns=['Dimension 1', 'Dimension 2'])
    df['Cluster'] = labels
    fig = px.scatter(
        df, x='Dimension 1', y='Dimension 2',
        color='Cluster',
        color_continuous_scale=px.colors.qualitative.G10,
        title=title,
        height=600
    )
    fig.update_layout(
        legend_title="Cluster",
        coloraxis_colorbar=dict(title="Cluster")
    )
    return fig


def plot_clusters_3d_plotly(data, labels, title="3D Cluster Visualization"):
    """Plot clusters in 3D using Plotly."""
    if data.shape[1] < 3:
        st.error("Need at least 3 dimensions for 3D visualization.")
        return None
    df = pd.DataFrame(data[:, :3], columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
    df['Cluster'] = labels
    fig = px.scatter_3d(
        df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
        color='Cluster',
        color_continuous_scale=px.colors.qualitative.G10,
        title=title,
        height=700
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        legend_title="Cluster",
        coloraxis_colorbar=dict(title="Cluster")
    )
    return fig


def plot_dendrogram_plotly(Z, labels=None, title="Hierarchical Clustering Dendrogram"):
    """Plot dendrogram using Plotly."""
    import scipy.cluster.hierarchy as sch

    fig = go.Figure()
    dendro = sch.dendrogram(Z, no_plot=True)

    x = []
    y = []
    for i, d in enumerate(dendro['dcoord']):
        x.extend([dendro['icoord'][i][j] for j in range(len(d))])
        x.append(None)
        y.extend(d)
        y.append(None)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='black', width=1.5),
        hoverinfo='none'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            title='Distance',
            showgrid=False,
            zeroline=False
        ),
        height=600,
        margin=dict(l=40, r=40, b=40, t=80)
    )
    return fig


# Function Definitions
def run_kmeans_clustering(data, n_clusters, distance_metric, use_3d=False):
    if data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return

    kmeans = KmeansClustering(k=n_clusters, distance=distance_metric)
    labels = kmeans.fit(data.values)

    if use_3d and data.shape[1] >= 3:
        fig = plot_clusters_3d_plotly(data.values, labels, f"K-Means Clustering (K={n_clusters}) - 3D View")
        st.plotly_chart(fig, use_container_width=True)
    else:
        try:
            fig = kmeans.plot_clusters(data.values, labels)
            st.pyplot(fig)
        except Exception:
            fig = plot_clusters_2d_plotly(data.values, labels, f"K-Means Clustering (K={n_clusters}) - 2D View")
            st.plotly_chart(fig, use_container_width=True)


def run_hierarchical_clustering(data, n_clusters_hier, linkage, use_3d=False):
    if data.shape[1] < 2:
        st.error("The dataset must contain at least two numeric columns for clustering.")
        return

    hierarchical_clustering = hierarchicalClustering(n_clusters=n_clusters_hier, linkage=linkage)
    labels = hierarchical_clustering.fit_predict(data.values)

    try:
        Z = hierarchical_clustering.Z  # Assuming Z is stored as an attribute
        fig_dendrogram = plot_dendrogram_plotly(Z, title="Hierarchical Clustering Dendrogram")
        st.plotly_chart(fig_dendrogram, use_container_width=True)
    except Exception:
        try:
            fig_dendrogram = hierarchical_clustering.plot_dendrogram()
            st.pyplot(fig_dendrogram)
        except Exception as e:
            st.error(f"Could not plot dendrogram: {e}")

    if use_3d and data.shape[1] >= 3:
        fig = plot_clusters_3d_plotly(data.values, labels, f"Hierarchical Clustering (K={n_clusters_hier}) - 3D View")
        st.plotly_chart(fig, use_container_width=True)
    else:
        try:
            fig_clusters = hierarchical_clustering.plot_clusters(data.values, labels)
            st.pyplot(fig_clusters)
        except Exception:
            fig = plot_clusters_2d_plotly(data.values, labels, f"Hierarchical Clustering (K={n_clusters_hier}) - 2D View")
            st.plotly_chart(fig, use_container_width=True)


def run_Elbow_Method(data):
    try:
        elbow = ElbowMethod(data, k_range=(1, 10))
        elbow.fit()
        optimal_k = elbow.find_optimal_k()
        fig = elbow.plot_elbow_curve()
        st.write(
            "La méthode du coude est une heuristique utilisée pour déterminer le nombre optimal de "
            "clusters dans un ensemble de données. La méthode consiste à tracer la variation expliquée "
            "en fonction du nombre de clusters, et à choisir le coude de la courbe comme nombre de "
            "clusters à utiliser."
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Nombre optimal de clusters : {optimal_k}")
        return optimal_k
    except AttributeError as e:
        st.error(f"Erreur dans ElbowMethod : {e}")
        st.write("Utilisation de la valeur par défaut de 3 clusters.")
        return 3


def show_kmeans_info():
    """Display information about K-means clustering."""
    st.markdown("""
## K-Means Clustering

### Description
K-Means est un algorithme de clustering par partitionnement qui divise un ensemble de données
en K groupes distincts en minimisant la variance intra-cluster.

### Fonctionnement
1. **Initialisation**: Sélection aléatoire de K centroïdes parmi les données
2. **Attribution**: Affectation de chaque point au centroïde le plus proche
3. **Mise à jour**: Recalculer la position des centroïdes en prenant la moyenne des points assignés
4. **Répétition**: Répéter les étapes 2 et 3 jusqu'à convergence

### Avantages
- Simple à comprendre et à implémenter
- Efficace et rapide pour les grands ensembles de données
- Fonctionne bien pour des clusters de forme sphérique

### Limites
- Nécessite de spécifier le nombre K de clusters à l'avance
- Sensible à l'initialisation des centroïdes
- Difficulté à gérer des clusters de formes irrégulières
- Sensibilité aux valeurs aberrantes

### Métriques de distance
- **Euclidienne**: Distance en ligne droite entre deux points
- **Manhattan**: Somme des différences absolues des coordonnées
- **Cosine**: Mesure l'angle entre deux vecteurs (similarité)
""")


def show_hierarchical_info():
    """Display information about hierarchical clustering."""
    st.markdown("""
## Classification Hiérarchique

### Description
La classification hiérarchique construit une hiérarchie de clusters, généralement représentée
sous forme d'arbre (dendrogramme).

### Fonctionnement
1. **Approche agglomérative (bottom-up)**:
   - Commencer avec chaque point comme cluster distinct
   - Fusionner progressivement les clusters les plus proches
   - Continuer jusqu'à n'avoir plus qu'un seul cluster
2. **Approche divisive (top-down)**:
   - Commencer avec un seul cluster contenant tous les points
   - Diviser récursivement les clusters
   - S'arrêter quand chaque cluster ne contient qu'un seul point

### Méthodes de liaison
- **Single linkage**: Distance entre les points les plus proches de deux clusters
- **Complete linkage**: Distance entre les points les plus éloignés
- **Average linkage**: Moyenne des distances entre tous les points
- **Ward**: Minimise la variance intra-cluster

### Avantages
- Pas besoin de spécifier le nombre de clusters à l'avance
- Fournit une représentation hiérarchique intuitive (dendrogramme)
- Peut découvrir des clusters de différentes formes et tailles

### Limites
- Complexité computationnelle élevée (O(n²) ou plus)
- Sensibilité au bruit et aux valeurs aberrantes
- Difficile à appliquer sur de très grands ensembles de données
""")


# Streamlit App Layout
st.title('Clustering Visualization')
st.write("by [Loubna Tildi](https://www.linkedin.com/in/loubna-tildi-4ba51a285/?locale=en_US)")

# Sidebar for clustering options
method = st.sidebar.radio(
    "Choose a clustering method:",
    ('K-Means Clustering', 'Hierarchical Clustering'),
    key='clustering_method'
)

# Add information section in sidebar
with st.sidebar.expander("Information sur les méthodes"):
    info_option = st.radio(
        "Sélectionnez une méthode pour en savoir plus:",
        ["K-Means", "Classification Hiérarchique"]
    )
    if info_option == "K-Means":
        show_kmeans_info()
    elif info_option == "Classification Hiérarchique":
        show_hierarchical_info()

# Add 3D visualization option
use_3d_viz = st.sidebar.checkbox("Utiliser la visualisation 3D", False, key='use_3d_viz')
if use_3d_viz and 'data' in st.session_state:
    if st.session_state.data.shape[1] < 3:
        st.sidebar.warning(
            "La visualisation 3D nécessite au moins 3 dimensions. "
            "Utilisez un jeu de données avec plus de colonnes."
        )
        use_3d_viz = False

n_clusters = None
distance_metric = None
linkage = "ward"  # Default linkage method

if method == 'K-Means Clustering':
    distance_metric = st.sidebar.selectbox(
        'Distance metric', ('Euclidean', 'Manhattan', 'Cosine'), key='kmeans_distance_metric'
    )
    use_optimal_k = st.sidebar.checkbox("Use optimal K from Elbow Method", True, key='use_optimal_k')
    if use_optimal_k:
        if 'optimal_k' not in st.session_state:
            st.session_state.optimal_k = 3
        n_clusters = st.session_state.optimal_k
    else:
        n_clusters = st.sidebar.slider(
            'Number of clusters', value=3, min_value=2, max_value=10, step=1, key='kmeans_n_clusters'
        )

n_clusters_hier = None
if method == 'Hierarchical Clustering':
    linkage = st.sidebar.selectbox(
        'Linkage method', ('ward', 'single', 'complete', 'average'), key='hier_linkage'
    )
    n_clusters_hier = st.sidebar.slider(
        'Number of clusters', value=3, min_value=2, max_value=10, step=1, key='hier_n_clusters'
    )

# Data input section
st.write("### Data Input:")
data_input_method = st.radio(
    "Choose how to input data:", ('Upload a file', 'Manual input', 'Random dataset'),
    key='data_input_method'
)

if data_input_method == 'Upload a file':
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)

elif data_input_method == 'Manual input':
    num_columns = st.number_input("Number of columns", min_value=1, value=3 if use_3d_viz else 2, step=1)
    column_names = [
        st.text_input(f"Column {i+1} name:", value=f"Feature {i+1}", key=f'manual_col_name_{i}')
        for i in range(num_columns)
    ]
    num_rows = st.number_input("Number of rows to generate", min_value=1, value=100, step=1)
    min_val, max_val = st.slider("Value range for random data", 0, 100, (0, 100))
    if st.button("Generate Data"):
        random_data = np.random.randint(min_val, max_val + 1, size=(num_rows, len(column_names)))
        st.session_state.data = pd.DataFrame(random_data, columns=column_names)

elif data_input_method == 'Random dataset':
    st.session_state.data = pd.read_csv('datasets/data1.txt', header=None)

# Ensure data preview and clustering execution only if data is available
if 'data' in st.session_state and not st.session_state.data.empty:
    st.write("#### Data Preview:")
    st.write(st.session_state.data)

    # Preprocess data
    numeric_data = st.session_state.data.apply(pd.to_numeric, errors='coerce').dropna()

    # Clustering execution
    if method == 'K-Means Clustering' and n_clusters and distance_metric:
        st.write("### Choosing the best K parameter for K-Means Clustering:")
        st.session_state.optimal_k = run_Elbow_Method(numeric_data)
        st.write("### K-Means Clustering:")
        run_kmeans_clustering(numeric_data, n_clusters, distance_metric, use_3d=use_3d_viz)

    elif method == 'Hierarchical Clustering' and n_clusters_hier:
        run_hierarchical_clustering(numeric_data, n_clusters_hier, linkage, use_3d=use_3d_viz)
