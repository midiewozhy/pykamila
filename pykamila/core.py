import numpy as np
from ._initialization import Initializer
from ._utils import split_cat_con, get_cat_levels
from ._continuous import distance_to_centroids, minimum_distances, radial_density, RadialTransform
from ._categorical import multinomial_probability
from ._kde import KernelDensityEstimator, MultinomialProbabilityEstimator
from ._assignment import assign_cluster
from ._update import update_centroids_theta, objective_function
from sklearn.preprocessing import OrdinalEncoder

class KAMILA:
    """
    The fundamental class for Kamila clustering.

    Parameters:
    ----------
    n_clusters: int
        The number of clusters specified by users.

    con_init: str
        The initialization method for continous variables.
        Default to be "default", can select from {'defult', 'kmenas++'}.

    cat_init: str
        The initialization method for categorical variables.
        Default to be "default", can select from {'default'}.

    n_init: int
        Number of times the algorithms is run with different seeds.
        Does not support n_init now, will upgrade later.

    max_iter: int
        The maximum number of iteration of the algorithm.
        Default to be 300.

    random_state: int or None
        Random seed for reproduction.
        Default to be None.

    Attributes:
    ----------
    centroids_: np.ndarray of shape (n_clusters, con_samples)
        The continous variable values for each cluster centroid.

    theta_: np.ndarray of shape (n_clusters, cat_samples, max_level)
        The categorical multinomial parameters for each cluster.

    labels_: np.ndarray of shape (n_samples, )
        Labels of each data point.

    obj_: float
        The sum of within cluster logged probability and density.

    """

    def __init__(self, 
                 n_clusters: int, 
                 con_init: str = 'default', 
                 cat_init: str = 'default', 
                 n_init: int = 10, 
                 max_iter: int = 300, 
                 random_state: int | None = None):

        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")

        if not isinstance(n_init, int) or n_init < 1:
            raise ValueError("n_init must be a positive integer.")
        
        self.n_clusters = n_clusters
        self.con_init = con_init
        self.cat_init = cat_init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids_ = None
        self.theta_ = None
        self.labels_ = None
        self.obj_ = None
        self.kde_ = None
        self.mpe_ = None
        self.rt_ = None
        self.encoder_ = None
        self.levels_ = None
        self.max_level_ = None
        self.con_idx_ = None
        self.scales_ = None

    def fit(self, X: np.ndarray, con_idx: np.ndarray, scales: np.ndarray = None, y=None):
        """
        Fit the clusterer.

        Parameters:
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The observed data.

        con_idx: np.ndarray of shape (con_features, )
            The array specifies which features are continuous.

        scales: np.ndarray of shape (con_features, )
            Scaling factor for each continuous variables.
            Default to be None

        y: Ignored
            Not used.

        Returns:
        ----------
        self: Kamila
            Fitted estimator.
        """

        # check if con_idx out of bounds
        if len(con_idx) == 0:
            raise ValueError("con_idx cannot be empty.")
        if con_idx.max() > X.shape[1] - 1:
            raise ValueError("Continuous feature index out of bounds for data array X.")

        # check if con_idx has correct type and 
        if not np.issubdtype(con_idx.dtype, np.integer):
            raise ValueError("Con_idx must have integer elements.")
        if np.unique(con_idx).size != con_idx.size:
            raise ValueError("Con_idx cannot have repeated elements.")
        
        # check scales matches con_idx
        if scales is not None and len(scales) != len(con_idx):
            raise ValueError("The length of scales and con_idx must match.")

        # store value for later use
        self.con_idx_ = con_idx
        self.scales_ = scales

        # split the data into continuous and categorical
        con, cat_raw = split_cat_con(X, self.con_idx_)

        if cat_raw.shape[1] == 0:
            raise ValueError("At least one categorical feature is required.")

        # encode categorical variables
        self.encoder_ = OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1)
        cat = self.encoder_.fit_transform(cat_raw).astype(np.int64)

        # get how many levels for each categorical variable, write in _utils later and max_level
        levels = get_cat_levels(cat)
        max_level = max(levels)

        # store values for later use
        self.levels_ = levels
        self.max_level_ = max_level

        best_obj = -np.inf
        
        for init in range(self.n_init):

            if self.random_state is not None:
                seed = self.random_state + init
            else:
                seed = None
        
            # initialize for centroids and theta
            centroids, theta = Initializer(self.n_clusters, self.con_init, self.cat_init, seed).initialize(con, cat, self.levels_, self.max_level_)
    
            # initialize jacobian transformation
            rt = RadialTransform(len(con_idx))
            
            # store for later use
            self.rt_ = rt
    
            # prev_labels to break from the algorithm if converge
            prev_labels = None
    
            # initialize estimators
            kde = KernelDensityEstimator()
            mpe = MultinomialProbabilityEstimator(self.levels_, self.max_level_)
            
            # iterative clustering
            for i in range(self.max_iter):
    
                # calculate distance to centroids and smallest distance
                dist = distance_to_centroids(con, centroids, scales)
                r = minimum_distances(dist)
    
                # calculate density and logged density of continuous variables
                kde.fit(r)
                fv, log_fv = radial_density(dist, kde, rt)
    
                # calculate probability and logged probability of categorical variables
                mpe.fit(theta)
                fc, log_fc = multinomial_probability(cat, mpe)
    
                # assign cluster label for each sample
                labels, max_p = assign_cluster(log_fv, log_fc)
    
                # update centroids and theta
                update_centroids_theta(con, cat, self.max_level_, labels, centroids, theta)
    
                # compute objective function value
                obj = objective_function(max_p)
    
                # stop criterion
                if prev_labels is not None and np.array_equal(prev_labels, labels):
                    break
                prev_labels = labels

            if obj > best_obj:
                best_obj = obj

                self.centroids_ = centroids
                self.theta_ = theta
                self.labels_ = labels
                self.obj_ = obj
                self.kde_ = kde
                self.mpe_ = mpe

        return self

    def predict(self, X: np.ndarray):
        """
        Predict for new data.

        Parameters:
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            New data to be clustered.

        Returns:
        labels: np.ndarray of shape (n_samples, )
            Labels of each data point.
        """

        if self.centroids_ is None:
            raise RuntimeError("The estimator is not fitted.")

        # parameters to be reused
        centroids, theta, con_idx, scales, levels = self.centroids_, self.theta_, self.con_idx_, self.scales_, self.levels_

        # class to be reused
        kde, mpe, rt = self.kde_, self.mpe_, self.rt_

        # split the data
        con, cat_raw = split_cat_con(X, con_idx)
        cat = self.encoder_.transform(cat_raw)
        
        # calculate distance to centroids and smallest distance
        dist = distance_to_centroids(con, centroids, scales)

        # calculate density and logged density of continuous variables
        fv, log_fv = radial_density(dist, kde, rt)

        # calculate probability and logged probability of categorical variables
        fc, log_fc = multinomial_probability(cat, mpe)

        # assign cluster label for each sample
        labels, _ = assign_cluster(log_fv, log_fc)

        return labels