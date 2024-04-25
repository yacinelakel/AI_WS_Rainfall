import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_embeddings(embedding_vectors, labels):
    # Create a PCA model
    pca_model = PCA(n_components=2, random_state=42)

    # Fit and transform the data to obtain PCA coordinates
    pca_result = pca_model.fit_transform(embedding_vectors)

    # Plot the PCA result with a different colormap and without color bar
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='plasma', s=10)  # Change the colormap to 'plasma'
    plt.title('PCA Projection of Embedding Vectors')
    plt.show()

import numpy as np

def l2_distance(embedding1, embedding2):
    """
    Compute the L2 distance between two embedding vectors.

    Parameters:
    - embedding1 (numpy array): First embedding vector.
    - embedding2 (numpy array): Second embedding vector.

    Returns:
    - float: L2 distance between the two embedding vectors.
    """
    return np.linalg.norm(embedding1 - embedding2)

def brute_force_knn(embeddings, query):
    similarities = embeddings.dot(query)
    sorted_ix = np.argsort(-similarities)
    return sorted_ix
