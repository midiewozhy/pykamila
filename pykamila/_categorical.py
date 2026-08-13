import numpy as np
from typing import Tuple
from _kde import MultinomialProbabilityEstimator


def multinomial_probability(
    sample: np.ndarray,
    mpe: MultinomialProbabilityEstimator
)-> Tuple[np.ndarray, np.ndarray] :
    """

    Compute the multinomial probability of categorical variables for
    each sample in each cluster.

    Parameters:
    ----------
    sample: ndarray of shape (n_samples, n_features)
        Categorical features for each sample.

    mpe: MultinomialProbabilityEstimator
        Multinomial probability estimator.

    Returns:
    ----------
    fc: ndarray of shape (n_samples, n_clusters)
        Multinomial probability for each sample in each cluster.

    log_fc: ndarray of shape (n_samples, n_clusters)
        Log of multinomial probabiliy for each sample in each cluster.

    """

    # calculate fc
    fc = mpe.evaluate(sample)

    # calculate log_fc
    log_fc = np.log(np.maximum(fc, np.finfo(float).eps)) # aviod log zero

    return fc, log_fc

    
