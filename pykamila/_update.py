import numpy as np
from typing import Tuple

def update_centroids_theta(con: np.ndarray, cat: np.ndarray, max_level: int, labels: np.ndarray, old_centroids: np.ndarray, old_theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the continous centroids and categorical parameters.

    Parameters:
    ----------
    con: np.ndarray of shape (n_samples, con_features)
        Array containing continous features.

    cat: np.ndarray of shape (n_samples, cat_features)
        Array containing categorical features.

    max_level: int
        The maximum level for all categorical variables.

    labels: np.ndarray of shape (n_samples, )
        Cluster label for each sample.

    old_centroids: np.ndarray of shape (n_clusters, con_features)
        old_centroids[i] is the estimated centroids for cluter
        i from previous iteration

    old_theta: np.ndarray of shape (n_clusters, cat_features, max_level)
        old_theta[k,j,l] is the estimated probability that
        categorical feature j takes level l in cluster k from
        the previous iteration.

    Return:
    ----------
    centroids: np.ndarray of shape (n_clusters, con_features)
        centroids[i] is the estimated centroid for cluster
        i.

    theta: np.ndarray of shape (n_clusters, cat_features, max_level)
        theta[k,j,l] is the estimated probability that
        categorical feature j takes level l in cluster k.

    """
    
    # count how many for each cluster and how many clusters
    count = np.bincount(labels)
    con_count = count[:, None]
    cat_count = count[:, None, None]
    n_clusters = len(count)

    # for continuous features
    # add up based on cluster label
    centroids = np.zeros((n_clusters, con.shape[1]))
    np.add.at(centroids, labels, con)

    # for categorical features
    # count and add up based on cluster label
    theta = np.zeros((n_clusters, cat.shape[1], max_level))

    rows = labels[:, None]
    cols = np.arange(cat.shape[1])[None, :]
    np.add.at(theta, (rows, cols, cat), 1)

    # calculating average and update
    np.divide(centroids, con_count, out = old_centroids, where = (con_count != 0))
    np.divide(theta, cat_count, out = old_theta, where = (cat_count != 0))
    

def objective_function(max_p: np.ndarray) -> float:
    """
    Compute the sum of the within cluster 
    logged proability and density.

    Parameters:
    ----------
    max_p: np.ndarray of shape (n_samples, )
        Within cluster logged probability and density

    Returns:
    ----------
    obj: float
        The sume of the within cluster logged probaility
        and density.

    """

    return np.sum(max_p)