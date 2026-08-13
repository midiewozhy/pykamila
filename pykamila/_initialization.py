import numpy as np
from typing import Tuple
from sklearn.cluster import kmeans_plusplus

class Initializer:
    """
    Initializer for cluster centroids.

    This class enables multi-type centroid initialization for mixed-type
    data.

    Attributes
    ----------
    n_clusters : int
        Number of centroids to initialize.

    con_type: str
        Indicate which specific initialization methods to use for continuous variables.

    cat_type: str
        Indicate which specific intialization methods to use for categorical parameters.

    random_state: int = 42
        Random seed for reproduction purpose.
        Default to be 42.
    """
    
    VALID_CON_INIT = ['default', 'kmeans++']
    VALID_CAT_INIT = ['default']

    def __init__(self,
                 n_clusters: int, 
                 con_type: str = "default", 
                 cat_type: str = "default", 
                 random_state: int = 42):

        # basic checking
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer.")
        if con_type not in self.VALID_CON_INIT:
            raise ValueError(f"con_type must be one of {self.VALID_CON_INIT}")
        if cat_type not in self.VALID_CAT_INIT:
            raise ValueError(f"cat_type must be one of {self.VALID_CAT_INIT}")
            
        self.n_clusters = n_clusters
        self.con_type = con_type
        self.cat_type = cat_type
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def initialize(self, con: np.ndarray, cat: np.ndarray, levs: np.ndarray, max_lev: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the centroids.

        Parameters:
        ----------
        con: np.ndarray of shape (n_samples, con_features)
            continuous sample data
            
        cat: np.ndarray of shape (n_samples, cat_features)
            categorical sample data, with categories transformed to 
            numerical levels

        levs: np.ndarray of shape (n_samples, )
            Array containing the number of levels for each categorical variables.

        max_lev: int
            The maximum number of level a categorical variable can have.

        Returns:
        ----------
        centroids: np.ndarray of shape (n_clusters, con_features)
            Array containing the centroids for continuous variables.
            
        theta: np.ndarray of shape (n_clusters, cat_features, max(n_levels))
            Categorical parameters.
            0 will be used to substitue if no
            such level exists for this feature.
            
        """

        # basic checking of data
        if con.ndim != 2 or cat.ndim != 2:
            raise ValueError("con and cat must be 2D arrays")
        if con.shape[0] != cat.shape[0]:
            raise ValueError("con and cat must have same number of samples")
        if con.shape[0] < self.n_clusters:
            raise ValueError("Number of samples must be >= n_clusters")
        if con.shape[1] == 0:
            raise ValueError("At least one continuous feature is required.")
        if cat.shape[1] == 0:
            raise ValueError("At least one categorical feature is required (or provide a dummy)")

        # basic checking for levs and max_lev
        if len(levs) != cat.shape[1]:
            raise ValueError("Length of levels must match number of categorical features.")
        if max_lev != max(levs):
            raise ValueError("max_level must equal max(levels).")
        if max_lev < 1:
            raise ValueError("max_level must be positive.")

        # initialize for continuous data
        if self.con_type == 'default':
            #Default initialization method is to sample
            #uniformly for each feature with the lower
            #and upper bound the minimum and the maximum
            #of the feature.

            # define the lower and upper bound
            col_min = np.min(con, axis = 0)
            col_max = np.max(con, axis = 0)

            # random sample
            centroids = self.rng.uniform(low = col_min, high = col_max, size = (self.n_clusters, con.shape[1]))

        elif self.con_type == 'kmeans++':
            centroids, _ = kmeans_plusplus(
                X=con, 
                n_clusters=self.n_clusters, 
                random_state=self.random_state
            )

        # initialize for categorical data
        if self.cat_type == 'default':
            #Default initialization method is to sample
            #from a Dirichlet distribution with shape
            #parameters all equal to one.

            num_of_features = cat.shape[1]

            # sample and stack
            theta = np.zeros((self.n_clusters, num_of_features, max_lev))
            for idx, lev in enumerate(levs):
                alpha = np.ones(lev)

                theta[:, idx, :lev] = self.rng.dirichlet(alpha = alpha, size = self.n_clusters)

        return centroids, theta
        
