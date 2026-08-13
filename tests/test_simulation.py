import numpy as np
import time
from scipy.stats import gennorm, norm, gaussian_kde
from scipy.optimize import root_scalar
from sklearn.metrics import adjusted_rand_score
from functools import lru_cache
from pykamila import KAMILA

# ---------- Empirical overlap calibration for lognormal ----------
def empirical_lognormal_overlap(shift_log, log_sd, n_calib=100000, random_state=42):
    rng = np.random.RandomState(random_state)
    n_clusters = 2
    n_con = 2
    pi = 0.5
    labels = rng.choice(n_clusters, size=n_calib, p=[pi, 1-pi])
    con = np.zeros((n_calib, n_con))
    
    for k in range(n_clusters):
        mask = (labels == k)
        log_mean = k * shift_log
        con[mask] = np.exp(rng.normal(loc=log_mean, scale=log_sd, size=(mask.sum(), n_con)))
        
    overlaps = []
    for j in range(n_con):
        x0 = con[labels == 0, j]
        x1 = con[labels == 1, j]
        grid = np.linspace(min(con[:, j]), max(con[:, j]), 2000)
        
        kde0 = gaussian_kde(x0)
        kde1 = gaussian_kde(x1)
        f0 = kde0(grid)
        f1 = kde1(grid)
        
        overlap = np.trapezoid(np.minimum(f0, f1), grid)
        overlaps.append(overlap)
    return np.mean(overlaps)

def calibrate_lognormal_shift(target_overlap, log_sd, bracket=(0.0, 30.0), tol=1e-4):
    def overlap_func(shift_log):
        return empirical_lognormal_overlap(shift_log, log_sd, n_calib=100000, random_state=42)

    sol = root_scalar(lambda s: overlap_func(s) - target_overlap,
                      bracket=bracket,
                      method='brentq',
                      xtol=tol)
    return sol.root

# ---------- Cached shift computation ----------
@lru_cache(maxsize=128)
def get_shift(distribution, overlap, p=None, log_sd=None):
    if distribution == 'normal':
        return 2 * norm.ppf(1 - overlap/2)
    elif distribution == 'pgnormal':
        return 2 * gennorm.ppf(1 - overlap/2, beta=p)
    elif distribution == 'lognormal':
        return calibrate_lognormal_shift(overlap, log_sd, bracket=(0.0, 30.0))

# ---------- Data generators ----------
def generate_pgnormal_mixture(n_samples=1000, n_con=2, n_cat=2, cat_levels=4,
                              con_overlap=0.30, cat_overlap=0.30, p=2.0, random_state=42):
    rng = np.random.RandomState(random_state)
    n_clusters = 2
    pi = 0.5

    shift = get_shift('pgnormal', con_overlap, p=p)
    labels = rng.choice(n_clusters, size=n_samples, p=[pi, 1-pi])
    con = np.zeros((n_samples, n_con))
    
    for k in range(n_clusters):
        mask = (labels == k)
        center = k * shift
        con[mask] = gennorm.rvs(beta=p, loc=center, scale=1.0,
                                size=(mask.sum(), n_con), random_state=rng)

    cat = np.zeros((n_samples, n_cat), dtype=int)
    eta = cat_overlap
    for j in range(n_cat):
        if cat_levels == 1:
            continue
        target0, target1 = 0, 1 % cat_levels
        for k in range(n_clusters):
            mask = (labels == k)
            prob = np.ones(cat_levels) * (eta / cat_levels)
            prob[target0 if k == 0 else target1] += 1 - eta
            prob /= prob.sum()
            cat[mask, j] = rng.choice(cat_levels, size=mask.sum(), p=prob)

    return np.hstack([con, cat]), labels, np.arange(n_con)

def generate_lognormal_mixture(n_samples=1000, n_con=2, n_cat=2, cat_levels=4,
                               con_overlap=0.30, cat_overlap=0.30, skew=1.0, random_state=42):
    rng = np.random.RandomState(random_state)
    n_clusters = 2
    pi = 0.5

    skew_to_log_sd = {1.0: 0.3143, 2.5: 0.6409, 9.0: 1.1310}
    log_sd = skew_to_log_sd.get(skew, 0.5)
    shift_log = get_shift('lognormal', con_overlap, log_sd=log_sd)

    labels = rng.choice(n_clusters, size=n_samples, p=[pi, 1-pi])
    con = np.zeros((n_samples, n_con))
    
    for k in range(n_clusters):
        mask = (labels == k)
        log_mean = k * shift_log
        con[mask] = np.exp(rng.normal(loc=log_mean, scale=log_sd, size=(mask.sum(), n_con)))

    cat = np.zeros((n_samples, n_cat), dtype=int)
    eta = cat_overlap
    for j in range(n_cat):
        if cat_levels == 1:
            continue
        target0, target1 = 0, 1 % cat_levels
        for k in range(n_clusters):
            mask = (labels == k)
            prob = np.ones(cat_levels) * (eta / cat_levels)
            prob[target0 if k == 0 else target1] += 1 - eta
            prob /= prob.sum()
            cat[mask, j] = rng.choice(cat_levels, size=mask.sum(), p=prob)

    return np.hstack([con, cat]), labels, np.arange(n_con)

# ---------- Simulation runner ----------
def run_simulation(data_generator, n_reps=50, n_init=10, **gen_kwargs):
    ari_list, times = [], []
    for rep in range(n_reps):
        X, true_labels, con_idx = data_generator(**gen_kwargs, random_state=rep+42)
        
        # Standardize continuous variables (z-score normalization)
        con = X[:, con_idx]
        con_std = con.std(axis=0)
        con_std[con_std == 0] = 1.0
        X[:, con_idx] = (con - con.mean(axis=0)) / con_std

        model = KAMILA(n_clusters=2, n_init=n_init, max_iter=100, random_state=rep+42)
        t0 = time.time()
        model.fit(X, con_idx)
        times.append(time.time() - t0)

        ari_list.append(adjusted_rand_score(true_labels, model.labels_))
        
    return np.mean(ari_list), np.std(ari_list), np.mean(times)


# ---------- Main ----------
if __name__ == "__main__":
    overlap_values = [0.01, 0.15, 0.30, 0.45]
    n_reps = 500               # for final paper-like accuracy
    n_init_pg = 10             # for p-generalized (easy)
    n_init_ln = 30             # for lognormal (hard) – more starts needed

    # ---- p-Generalized Normal (kurtosis=6) ----
    print("\n" + "="*70)
    print("SIMULATION A: p-GENERALIZED NORMAL (kurtosis=6, p=0.7785)")
    print("="*70)
    print(f"{'Con Ov':<8} {'Cat Ov':<8} {'ARI Mean':<12} {'ARI Std':<12} {'Time (s)':<10}")
    print("-"*70)
    for con_ov in overlap_values:
        for cat_ov in overlap_values:
            shift = get_shift('pgnormal', con_ov, p=0.7785)
            print(f"  Shift for con_overlap={con_ov:.2f}: {shift:.4f}")
            mean_ari, std_ari, avg_time = run_simulation(
                generate_pgnormal_mixture,
                n_reps=n_reps,
                n_init=n_init_pg,
                n_samples=1000,
                n_con=2,
                n_cat=2,
                cat_levels=4,
                con_overlap=con_ov,
                cat_overlap=cat_ov,
                p=0.7785
            )
            print(f"{con_ov:<8.2f} {cat_ov:<8.2f} {mean_ari:<12.3f} {std_ari:<12.3f} {avg_time:<10.3f}")

    # ---- Lognormal (skewness=9) ----
    print("\n" + "="*70)
    print("SIMULATION A: LOGNORMAL (skewness=9.0)")
    print("="*70)
    print(f"{'Con Ov':<8} {'Cat Ov':<8} {'ARI Mean':<12} {'ARI Std':<12} {'Time (s)':<10}")
    print("-"*70)
    for con_ov in overlap_values:
        for cat_ov in overlap_values:
            shift_log = get_shift('lognormal', con_ov, log_sd=1.1310)
            print(f"  Shift_log for con_overlap={con_ov:.2f}: {shift_log:.4f}")
            mean_ari, std_ari, avg_time = run_simulation(
                generate_lognormal_mixture,
                n_reps=n_reps,
                n_init=n_init_ln,        # more starts for skewed data
                n_samples=1000,
                n_con=2,
                n_cat=2,
                cat_levels=4,
                con_overlap=con_ov,
                cat_overlap=cat_ov,
                skew=9.0
            )
            print(f"{con_ov:<8.2f} {cat_ov:<8.2f} {mean_ari:<12.3f} {std_ari:<12.3f} {avg_time:<10.3f}")