import numpy as np
from typing import Tuple

def assign_cluster(log_fv: np.ndarray, log_fc: np.ndarray) -> Tuple[np.ndarray, np.ndarray] :
    """
    Assign cluster membership to each sample.

    Parameters:
    ----------
    log_fv: np.ndarray of shape (n_samples, n_clusters)
        Array containing logged density of each sample
        to each cluster centroid for continuous variables.

    log_fc: np.ndarray of shape (n_samples, n_clusters)
        Array containing logged probability of each sample
        based on categorical multinomial distribution parameters.

    Returns:
    ----------
    labels: np.ndarray of shape (n_samples, )
        Cluster label for each sample.

    max_p: np.ndarray of shape (n_samples, )
        Maximum logged probability and density
        for each sample.
        
    """

    # add up the logged density/probability
    log_p = log_fv + log_fc

    # find the maximum and get the index
    labels = np.argmax(log_p, axis = 1)

    # get the maximum logged value
    max_p = np.max(log_p, axis = 1)

    return labels, max_p