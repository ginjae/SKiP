import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

def precompute_class_stats(X_train, y_train, percentile=99):
    """
    각 클래스별로:
      - mean (mu)
      - covariance (cov)
      - inverse covariance (inv_cov)
      - Mahalanobis distance의 percentile-th threshold (tau)
    를 한 번씩만 계산해서 캐싱.
    """
    class_stats = {}
    classes = np.unique(y_train)

    for cls in classes:
        X_cls = X_train[y_train == cls]

        mu = X_cls.mean(axis=0)
        cov = np.cov(X_cls.T) + np.eye(X_train.shape[1]) * 1e-6
        inv_cov = np.linalg.inv(cov)

        diffs = X_cls - mu
        dists = np.einsum('ni,ij,nj->n', diffs, inv_cov, diffs)

        tau = np.percentile(dists, percentile)

        class_stats[cls] = {
            "mu": mu,
            "cov": cov,
            "inv_cov": inv_cov,
            "tau": tau,
        }

    return class_stats


def inject_noise(X_train, y_train, feature_noise=0.0, label_noise=0.0, random_state=42):
    """
    Injects Type I (feature) and Type II (label) outliers into X_train, y_train.
    
    Type I Outlier:
        - For each class y, compute mean μ_y and covariance Σ_y.
        - Generate points far from the class distribution using:
              (x' - μ_y)^T Σ_y^{-1} (x' - μ_y) > τ
        - τ is chosen as the 99th percentile of Mahalanobis distances.

    Type II Outlier:
        - Random label flips.

    Parameters:
        X_train (np.ndarray): Feature matrix
        y_train (np.ndarray): Labels
        feature_noise (float): Ratio of synthetic feature outliers
        label_noise (float): Ratio of label flips
        random_state (int): Seed for reproducibility

    Returns:
        X_noisy, y_noisy
    """

    rng = default_rng(random_state)
    X = X_train.copy()
    y = y_train.copy()

    # ---------------------------------------------
    # 1. Type II Outliers (Label Noise)
    # ---------------------------------------------
    if label_noise > 0:
        n_label_flips = int(len(y) * label_noise)
        flip_indices = rng.choice(len(y), n_label_flips, replace=False)
        unique_labels = np.unique(y)

        for idx in flip_indices:
            original = y[idx]
            y[idx] = rng.choice(unique_labels[unique_labels != original])

    # ---------------------------------------------
    # 2. Type I Outliers (Feature Noise)
    # ---------------------------------------------
    if feature_noise > 0:
        class_stats = precompute_class_stats(X_train, y_train, percentile=99)

        n_feature_outliers = int(len(X) * feature_noise)
        outliers = []
        outlier_labels = []

        classes = np.array(list(class_stats.keys()))

        for _ in range(n_feature_outliers):
            cls = rng.choice(classes)
            stats = class_stats[cls]
            mu = stats["mu"]
            cov = stats["cov"]
            inv_cov = stats["inv_cov"]
            tau = stats["tau"]

            while True:
                x_candidate = rng.multivariate_normal(mu, cov)
                diff = x_candidate - mu
                d = diff @ inv_cov @ diff  # Mahalanobis distance

                if d > tau:
                    outliers.append(x_candidate)
                    outlier_labels.append(cls)
                    break

        outliers = np.array(outliers)
        outlier_labels = np.array(outlier_labels)

        X = np.vstack([X, outliers])
        y = np.concatenate([y, outlier_labels])

    return X, y
