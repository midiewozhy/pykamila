import numpy as np
from typing import Tuple

# preprocesssing function
def split_cat_con(data: np.ndarray, con_index: np.ndarray)-> Tuple[np.ndarray, np.ndarray] :
    """
    Split the data set into categorical subset and continuous subset.

    Parameters:
    ----------
    data: np.ndarray of shape (n_samples, total_features)
        sample data
        
    con_index: np.ndarray of shape (con_features,)
        numpy array contains the index of the continuous variables

    Returns:
    ----------
    con: np.ndarray of shape (n_samples, con_features)
        continuous sample data
        
    cat: np.ndarray of shape (n_samples, cat_features)
        categorical sample data, with categories transformed to 
        numerical levels
        
    """

    # split the data into continuous and categorical
    con = data[:, con_index]

    total_features = data.shape[1]
    cat_mask = np.ones(total_features, dtype=bool)
    cat_mask[con_index] = False
    cat_raw = data[:, cat_mask]
    
    return con, cat_raw

def get_cat_levels(cat: np.ndarray) -> np.ndarray:
    """
    Get the levels for each categorical variables.

    Parameters:
    ----------
    cat: np.ndarray of shape (n_samples, cat_features)
        Sample categorical data.

    Returns:
    ----------
    levels: np.ndarray of shpe (cat_features, )
        Array containing the level for each categorical variable.

    """

    return np.array([len(np.unique(cat[:, col])) for col in range(cat.shape[1])])