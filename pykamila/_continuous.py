import numpy as np
import math
from typing import Tuple
from ._kde import KernelDensityEstimator

def distance_to_centroids(
    X: np.ndarray,
    centroids: np.ndarray,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the scaled Euclidean distance from every sample to every centroid.

    Parameters
    ----------
    X : sample data, np.ndarray of shape (n_samples, n_features)

    centroids : centroids matrix, np.ndarray of shape (n_clusters, n_features)

    scales : scale factor, np.ndarray of shape (n_features,), optional; act as weight but is actually scaling factor

    Returns
    -------
    dist : distance matrix, np.ndarray of shape (n_samples, n_clusters)

        distances[i, j] is the scaled Euclidean distance
        between sample i and centroid j.
    """

    # basic checking
    if scales is not None:
        if scales.ndim != 1 or scales.shape[0] != X.shape[1]:
            raise ValueError("scales must be a 1D array of length n_features.")
        if np.any(scales < 0):
            raise ValueError("scales must be non-negative.")
    
    # compute diff matrix
    diff = X[:, np.newaxis, :] - centroids[np.newaxis ,:, :]
    if scales is not None:
        s = scales[np.newaxis, np.newaxis, :]
        diff *= s
        
    # compute dist matrix
    dist_sqr = np.sum(np.square(diff), axis=2)
    dist = np.sqrt(dist_sqr)

    return dist

def minimum_distances(
    dist: np.ndarray
) -> np.ndarray:
    """
    Compute the minimum distance from each sample to its nearest centroid.

    Parameters
    ----------
    dist: distance matrix, np.ndarray of shape (n_samples, n_clusters)

    Returns
    ----------
    r: minimum distances, np.ndarray of shape (n_samples,)

        r[i] is the smallest distance between sample i
        and all centroids.
    """

    r = np.min(dist, axis=1)

    return r

class RadialTransform:
    """
    Transform a radial density into the corresponding multivariate density.

    This class encapsulates the dimension-dependent factor in Equation (2)
    of the KAMILA algorithm. Given a distance from a cluster centroid, it
    evaluates the Jacobian term

        Γ(P/2 + 1)
        -----------------
        P π^(P/2) r^(P-1)

    where P is the number of continuous features.

    Parameters
    ----------
    n_features : int
        Number of continuous features (P).

    Attributes
    ----------
    n_features : int
        Number of continuous features.

    exponent : int
        Exponent P - 1 applied to the distance.

    coef : float
        Dimension-dependent constant

            Γ(P/2 + 1)
            ----------
             P π^(P/2)

        used in the density transformation.
    """
    def __init__(self, n_features: int):
        self.n_features = n_features
        self.exponent = n_features - 1
        self.coef = math.gamma(1+n_features/2) / (n_features * math.pi ** (n_features/2))

    def jacobian(self, dist: np.ndarray) -> np.ndarray:
        """
        Evaluate the Jacobian term for the given distances.

        Parameters
        ----------
        dist : np.ndarray
            Distance(s) from observations to cluster centroids. (n_samples, n_clusters)

        Returns
        -------
        jacobian : np.ndarray of shape (n_samples, n_clusters)
            Jacobian values with the same shape as ``dist``.
        """
        
        # avoid divide by 0
        eps = np.finfo(float).eps

        # get the nonzero dist
        r = np.maximum(dist, eps)

        # compute the jacobian
        jacobian = self.coef / np.power(r, self.exponent)

        return jacobian

def radial_density(
    dist: np.ndarray,
    kde: KernelDensityEstimator,
    transform: RadialTransform
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the radial density corresponding to the fitted KDE
    at each sample-centroid distance.

    Parameters
    ----------
    dist : np.ndarray of shape (n_samples, n_clusters)
        Distance from every sample to every centroid.

    kde : KernelDensityEstimator
        Kernel density estimator fitted on the minimum distances.

    transform: RadialTransform
        RadialTransform used to calculate jacobian.

    Returns
    -------
    fv : np.ndarray of shape (n_samples, n_clusters)
        Radial density evaluated at every distance.
        
    log_fv: np.ndarray of shape (n_samples, n_clusters)
        Log of Radial density evaluated at every distance.
    """
    # calculate f_R(d_ig)
    fr = kde.evaluate(dist)
    
    # calculate f_V(d_ig)
    fv = fr * transform.jacobian(dist)

    # calculate the log
    log_fv = np.log(np.maximum(fv, np.finfo(float).eps))

    # return the values
    return fv, log_fv

