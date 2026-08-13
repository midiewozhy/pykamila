import pytest
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import OrdinalEncoder
from pykamila import KAMILA
from pykamila._utils import split_cat_con, get_cat_levels
from pykamila._initialization import Initializer
from pykamila._continuous import RadialTransform
from pykamila._kde import KernelDensityEstimator, MultinomialProbabilityEstimator

# --------------------- Helper functions for synthetic data ---------------------

def generate_mixed_data(n_samples=500, n_clusters=2, n_con=2, n_cat=2, cat_levels=4,
                        con_sep=3.0, cat_sep=0.8, random_state=42):
    """
    Generate a synthetic mixed dataset with known cluster labels.

    Continuous variables: Gaussian mixtures with means separated by con_sep.
    Categorical variables: Multinomial mixtures with probabilities differing by cat_sep.

    Returns:
        X: np.ndarray of shape (n_samples, n_con + n_cat)
        labels: true cluster labels
        con_idx: indices of continuous columns
    """
    rng = np.random.RandomState(random_state)

    # Assign each sample to a cluster
    cluster_probs = np.ones(n_clusters) / n_clusters
    labels = rng.choice(n_clusters, size=n_samples, p=cluster_probs)

    # Continuous features: Gaussian means
    con_data = np.zeros((n_samples, n_con))
    for k in range(n_clusters):
        mask = (labels == k)
        # centers form a grid or simple separation
        center = np.full(n_con, k * con_sep)
        con_data[mask] = rng.normal(loc=center, scale=1.0, size=(mask.sum(), n_con))

    # Categorical features: multinomial with different probabilities per cluster
    cat_data = np.zeros((n_samples, n_cat), dtype=int)
    for k in range(n_clusters):
        mask = (labels == k)
        for j in range(n_cat):
            # base probability for level 0
            base_prob = np.ones(cat_levels) / cat_levels
            # shift probability toward level (k % cat_levels) by cat_sep
            if cat_levels > 1:
                target = k % cat_levels
                prob = np.ones(cat_levels) * (1 - cat_sep) / (cat_levels - 1)
                prob[target] = cat_sep
                # ensure sum to 1
                prob = prob / prob.sum()
            else:
                prob = np.array([1.0])
            cat_data[mask, j] = rng.choice(cat_levels, size=mask.sum(), p=prob)

    # Combine data: continuous first, then categorical
    X = np.hstack([con_data, cat_data])
    con_idx = np.arange(n_con)
    return X, labels, con_idx


def test_kamila_basic_fit():
    """Test that KAMILA runs and returns labels of correct shape."""
    X, true_labels, con_idx = generate_mixed_data(n_samples=200, n_clusters=2)
    model = KAMILA(n_clusters=2, max_iter=100, random_state=42)
    model.fit(X, con_idx)
    assert model.labels_.shape == (X.shape[0],)
    assert model.centroids_.shape == (2, len(con_idx))
    assert model.theta_.shape == (2, X.shape[1] - len(con_idx), max(model.levels_))
    assert model.obj_ is not None


def test_kamila_clustering_accuracy():
    """Test that KAMILA recovers clusters reasonably well on separable data."""
    X, true_labels, con_idx = generate_mixed_data(n_samples=500, n_clusters=2,
                                                  con_sep=4.0, cat_sep=0.9)
    model = KAMILA(n_clusters=2, max_iter=100, random_state=42)
    model.fit(X, con_idx)
    ari = adjusted_rand_score(true_labels, model.labels_)
    # Expect near-perfect separation
    assert ari > 0.9


def test_kamila_predict():
    """Test predict method on new data."""
    X, true_labels, con_idx = generate_mixed_data(n_samples=300, n_clusters=2)
    # Fit on first 200 samples
    X_train = X[:200]
    X_test = X[200:]
    model = KAMILA(n_clusters=2, max_iter=100, random_state=42)
    model.fit(X_train, con_idx)
    pred_labels = model.predict(X_test)
    assert pred_labels.shape == (X_test.shape[0],)
    # Check that predictions are consistent with training (ARI on full set)
    full_labels = np.concatenate([model.labels_, pred_labels])
    ari = adjusted_rand_score(true_labels, full_labels)
    assert ari > 0.75


def test_kamila_scales():
    """Test that providing scales works."""
    X, true_labels, con_idx = generate_mixed_data(n_samples=200, n_clusters=2)
    # Use arbitrary scales
    scales = np.ones(len(con_idx)) * 0.5
    model = KAMILA(n_clusters=2, max_iter=50, random_state=42)
    model.fit(X, con_idx, scales=scales)
    assert model.labels_.shape == (X.shape[0],)


def test_kamila_init_methods():
    """Test that different initialization methods run without error."""
    X, _, con_idx = generate_mixed_data(n_samples=100, n_clusters=3)
    for con_init in ['default', 'kmeans++']:
        model = KAMILA(n_clusters=3, con_init=con_init, max_iter=20, random_state=42)
        model.fit(X, con_idx)
        assert model.labels_.shape == (X.shape[0],)


def test_kamila_single_cluster():
    """Test edge case: n_clusters=1."""
    X, _, con_idx = generate_mixed_data(n_samples=100, n_clusters=1)  # true clusters=1 but we set n_clusters=1
    model = KAMILA(n_clusters=1, max_iter=20, random_state=42)
    model.fit(X, con_idx)
    assert np.all(model.labels_ == 0)
    assert model.centroids_.shape == (1, len(con_idx))


def test_kamila_insufficient_samples():
    """Test that fitting fails if samples < n_clusters."""
    X, _, con_idx = generate_mixed_data(n_samples=5, n_clusters=10)  # more clusters than samples
    model = KAMILA(n_clusters=10, max_iter=10)
    with pytest.raises(ValueError, match="Number of samples must be >= n_clusters"):
        model.fit(X, con_idx)


def test_kamila_convergence():
    """Test that algorithm stops within max_iter and that labels stabilize."""
    X, _, con_idx = generate_mixed_data(n_samples=300, n_clusters=2)
    model = KAMILA(n_clusters=2, max_iter=10, random_state=42)
    model.fit(X, con_idx)
    # If it converged, max_iter should not be reached? We can check if iteration count is less than max_iter
    # But we don't have iteration count exposed. We can check that labels are stable: run again and compare.
    model2 = KAMILA(n_clusters=2, max_iter=10, random_state=42)
    model2.fit(X, con_idx)
    # Since random_state fixed, should get same labels
    ari = adjusted_rand_score(model.labels_, model2.labels_)
    assert ari == 1.0


def test_kamila_categorical_varying_levels():
    """Test with categorical variables having different numbers of levels."""
    X, true_labels, con_idx = generate_mixed_data(n_samples=200, n_clusters=2, n_cat=2, cat_levels=5)
    # Modify one categorical to have only 2 levels
    X[:, con_idx.shape[0] + 1] = X[:, con_idx.shape[0] + 1] % 2
    model = KAMILA(n_clusters=2, max_iter=50, random_state=42)
    model.fit(X, con_idx)
    assert model.levels_[1] == 2  # second categorical variable should have 2 levels
    assert model.theta_.shape[2] == max(model.levels_)  # max level
    # Test predict on new data
    X_new, _, _ = generate_mixed_data(n_samples=50, n_clusters=2, n_cat=2, cat_levels=5)
    X_new[:, con_idx.shape[0] + 1] = X_new[:, con_idx.shape[0] + 1] % 2
    pred = model.predict(X_new)
    assert pred.shape == (50,)


def test_kamila_no_continuous():
    """Test that error is raised if no continuous features are provided."""
    X, _, con_idx = generate_mixed_data(n_samples=100, n_con=0)  # no continuous
    # con_idx empty
    with pytest.raises(ValueError, match="con_idx cannot be empty"):
        model = KAMILA(n_clusters=2)
        model.fit(X, con_idx)


def test_kamila_no_categorical():
    X, _, con_idx = generate_mixed_data(n_samples=100, n_cat=0)
    model = KAMILA(n_clusters=2, max_iter=10, random_state=42)
    with pytest.raises(ValueError, match="At least one categorical feature is required."):
        model.fit(X, con_idx)


def test_kamila_estimator_not_fitted_predict():
    """Test predict raises error if not fitted."""
    X, _, con_idx = generate_mixed_data(n_samples=100, n_clusters=2)
    model = KAMILA(n_clusters=2)
    with pytest.raises(RuntimeError, match="The estimator is not fitted"):
        model.predict(X)


def test_kamila_multiple_init_not_implemented():
    """Test that n_init > 1 raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Multiple init has not be implemented yet"):
        KAMILA(n_clusters=2, n_init=3)


# --------------------- Integration with utilities ---------------------
def test_utils_split_cat_con():
    """Test the split_cat_con utility."""
    X, _, con_idx = generate_mixed_data(n_samples=10, n_con=2, n_cat=3)
    con, cat = split_cat_con(X, con_idx)
    assert con.shape == (10, 2)
    assert cat.shape == (10, 3)
    # Check that concatenation reconstructs X
    combined = np.hstack([con, cat])
    np.testing.assert_array_equal(combined, X)


def test_utils_get_cat_levels():
    """Test getting categorical levels."""
    X, _, con_idx = generate_mixed_data(n_samples=1000, n_con=1, n_cat=3,
                                        cat_levels=4, random_state=42)
    _, cat = split_cat_con(X, con_idx)
    levels = get_cat_levels(cat)
    # With enough samples, all levels should appear
    np.testing.assert_array_equal(levels, np.array([4, 4, 4]))


def test_initializer_shapes():
    """Test that initializer returns correct shapes."""
    X, true_labels, con_idx = generate_mixed_data(n_samples=200, n_clusters=3, n_con=2, n_cat=2, cat_levels=4)
    con, cat = split_cat_con(X, con_idx)
    levels = get_cat_levels(cat)
    max_level = max(levels)
    init = Initializer(n_clusters=3, random_state=42)
    centroids, theta = init.initialize(con, cat, levels, max_level)
    assert centroids.shape == (3, con.shape[1])
    assert theta.shape == (3, cat.shape[1], max_level)
    # Check that probabilities sum to 1 for each categorical variable and cluster
    for k in range(3):
        for j in range(cat.shape[1]):
            assert np.isclose(theta[k, j, :levels[j]].sum(), 1.0)
            assert np.all(theta[k, j, levels[j]:] == 0)  # beyond levels are zero


def test_radial_transform():
    """Test RadialTransform Jacobian."""
    rt = RadialTransform(n_features=3)
    dist = np.array([[1.0, 2.0], [0.5, 1.5]])
    jac = rt.jacobian(dist)
    assert jac.shape == (2, 2)
    # At r=0, jacobian should be finite (clipped)
    dist_zero = np.array([[0.0]])
    jac_zero = rt.jacobian(dist_zero)
    assert np.isfinite(jac_zero[0,0])


def test_kde_estimator():
    """Test KDE estimator fit and evaluate."""
    r = np.array([0.5, 1.0, 1.5, 2.0, 0.8])
    kde = KernelDensityEstimator()
    kde.fit(r)
    dist = np.array([[0.5, 1.0], [1.2, 2.5]])
    density = kde.evaluate(dist)
    assert density.shape == (2, 2)
    assert np.all(density >= 0)


def test_multinomial_prob_estimator():
    """Test MultinomialProbabilityEstimator."""
    levels = np.array([3, 2])
    max_level = 3
    theta = np.zeros((2, 2, max_level))
    # cluster 0: first variable probs [0.8,0.1,0.1], second [0.9,0.1,0]
    theta[0,0,:] = [0.8,0.1,0.1]
    theta[0,1,:] = [0.9,0.1,0.0]
    # cluster 1: [0.2,0.3,0.5], [0.4,0.6,0.0]
    theta[1,0,:] = [0.2,0.3,0.5]
    theta[1,1,:] = [0.4,0.6,0.0]
    mpe = MultinomialProbabilityEstimator(levels, max_level)
    mpe.fit(theta)
    sample = np.array([[0, 1], [2, 0]])  # shape (2,2)
    prob = mpe.evaluate(sample)
    # Expected: for sample[0] cluster0: 0.8*0.1=0.08; cluster1:0.2*0.6=0.12
    # sample[1] cluster0: 0.1*0.9=0.09; cluster1:0.5*0.4=0.20
    expected = np.array([[0.08, 0.12], [0.09, 0.20]])
    np.testing.assert_allclose(prob, expected)


if __name__ == "__main__":
    pytest.main([__file__])