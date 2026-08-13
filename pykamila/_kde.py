import numpy as np
import math
import scipy.stats as sp

# main class
class KernelDensityEstimator:
    """
    One-dimensional kernel density estimator.

    This class estimates the probability density of the radial distances
    using a kernel density estimator (KDE). The fitted estimator can then
    be evaluated at arbitrary distances.

    Attributes
    ----------
    n_samples : int
        Number of observations used to fit the estimator.

    bandwidth_type: str
        Name of the bandwidth used.

    r : np.ndarray of shape (n_samples,)
        Training sample.

    kernel_type : str
        Name of the kernel function.
    """
    def __init__(self):
        self.r = None
        self.bandwidth_type = None
        self.kernel_type = None
        self.n_samples = None
        self.kernel = None

    def fit(self, r: np.ndarray, bandwidth_type: str | int | float = 'silverman', kernel_type: str = 'gaussian'):
        """
        Fit the kernel density estimator.
    
        This method validates the specified kernel type and initializes
        the corresponding kernel function for subsequent density evaluation.

        Parameters
        ----------
        r: np.ndarray of shape (n_samples, )
            Minimum distances from each sample to its nearest centroids.

        bandwidth_type: str or int or float, default "silverman"
            Selected bandwidth type used to smooth out kernel estimation.
            If it is a string, then can only be from {"silverman", "scott", "silverman_rule_of_thumb"}.

        kernel_type: str, default "gaussian"
            Selected kernel type used to estimate density.
            Only support gaussian kernel now.
    
        Returns
        ----------
        self : KernelDensityEstimator
            Fitted estimator.
        """

        # check dimension of r
        if r is None or r.ndim != 1 or r.shape[0] == 0:
            raise ValueError("r must be nonempty 1D array.")

        # check the bandwidth_type
        if isinstance(bandwidth_type, str):
            if bandwidth_type not in {"silverman", "scott", "silverman_rule_of_thumb"}:
                raise ValueError("can only use silverman, silverman_rule_of_thumb or scott bandwidth")
        elif not isinstance(bandwidth_type, (float, int)):
            raise ValueError("the bandwidth factor can only be silverman, scott, silverman_rule_of_thumb or a scalar.")

        # check the kernel_type
        if kernel_type not in {'gaussian'}:
            raise NotImplementedError("only supports gaussian kernel.")

        self.r = r
        self.bandwidth_type = bandwidth_type
        self.kernel_type = kernel_type
        self.n_samples = r.shape[0]

        if self.bandwidth_type == "silverman_rule_of_thumb":
            # later implement the edge case for sigma = 0
            q75, q25 = np.quantile(self.r, [0.75, 0.25])
            iqr = q75 - q25
            sigma = np.std(self.r, ddof = 1)
            A = min(iqr / 1.34, sigma) if iqr > 0  else sigma
            bw = 0.9 * A * (self.n_samples ** (-0.2))
            self.kernel = sp.gaussian_kde(self.r, bw_method = bw / sigma)
        else:
            self.kernel = sp.gaussian_kde(self.r, bw_method = self.bandwidth_type)

        return self

    def evaluate(self, dist: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted kernel density estimator.
    
        Parameters
        ----------
        dist : np.ndarray of shape (n_samples, n_clusters)
            Distance from each observation to every cluster centroid.
    
        Returns
        ----------
        density : np.ndarray of shape (n_samples, n_clusters)
            Estimated radial density evaluated at each distance.
        """
        
        # check if estimator is fitted
        if self.kernel is None:
            raise RuntimeError("KernelDensityEstimator has not been fitted.")

        # check dimension of dist
        if dist is None or dist.ndim != 2:
            raise ValueError("dist must be a nonempty 2D array.")
        
        # edge case
        if dist.size == 0:
            return np.empty((0, dist.shape[1]))

        return self.kernel.evaluate(dist.ravel()).reshape(dist.shape)


class MultinomialProbabilityEstimator:
    """
    Multinomial probability estimator for categorical variables.

    This class estimates the probability mass of categorical variables
    under the conditional independence assumption of categorical features
    in one cluster.

    Attributes:
    ----------
    theta: np.ndarray of shape (n_clusters, n_features, max(n_levels))
        Multinomial distribution parameters.
        0 will be used to substitue if no
        such level exists for this feature.

    n_clusters: int
        Number of clusters.

    n_features: int
        Number of categorical variables.

    levels: np.ndarray of shape (n_features, )
        levels[i] is the total number of levels
        for the i-th categorical feature.

    max_level: int
        The maximum levels of categorical variables.
     
    """
    def __init__(self, levels: np.ndarray, max_level:int):
        """
        Initialze the class.

        Parameters:
        ----------
        levels: np.ndarray of shape (n_features,)
            levels[i] is the total number of levels
            for the i-th categorical feature.

        max_level: int
            The maximum number of level a categorical variable can have.
        """
        self.theta = None
        self.n_clusters = None
        self.n_features = levels.shape[0]
        self.levels = levels
        self.max_level = max_level

    def fit(self, theta: np.ndarray):
        """
        Fit the probability estimator.

        Parameters:
        ----------
        theta: np.ndarray of shape (n_clusters, n_features, max(n_levels))
            Multinomial distribution parameters.
            0 will be used to substitue if no
            such level exists for this feature.

        Returns
        ----------
        self : MultinomialProbabilityEstimator
            Fitted estimator.
        """
        # check dimension of theta
        if theta.shape[1:] != (self.n_features, self.max_level):
            raise ValueError(f"theta shape must be (n_clusters, {self.n_features}, {self.max_level})")

        self.theta = theta
        self.n_clusters = theta.shape[0]

        return self

    def evaluate(self, sample: np.ndarray) -> np.ndarray:
        """
        Evaluated the fitted multinomial probability estimator.

        Parameters:
        ----------
        sample: np.ndarray of shape (n_samples, n_features)
            Data containing all categorical features for every sample.

        Returns:
        ----------
        fc: np.ndarray of shape (n_samples, n_clusters)
            Estimated probability for each sample in each cluster
        """

        # check if estimator is fitted
        if self.theta is None:
            raise RuntimeError("Multinomial Probability Estimator is not fitted.")

        # check dimension of sample data
        if sample.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {sample.shape[1]}")

        # clip the sample, so that no unseen level during training (encodede by -1) will be used for proabability extraction
        clipped_sample = np.clip(sample, 0, self.max_level - 1).astype(np.int64)

        # compute multinomial probability
        cluster_idx = np.arange(self.n_clusters).reshape(1, -1, 1)
        feature_idx = np.arange(self.n_features).reshape(1, 1, -1)
        level_idx = clipped_sample[:, None, :]

        prob = self.theta[cluster_idx, feature_idx, level_idx]

        invalid_mask = (sample < 0) | (sample >= self.levels[None, :])
        invalid_mask_expanded = np.broadcast_to(invalid_mask[:, np.newaxis, :], prob.shape)
        prob[invalid_mask_expanded] = 0.0
        
        fc = prob.prod(axis = 2)

        return fc