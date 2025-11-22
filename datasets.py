import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

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
        n_feature_outliers = int(len(X) * feature_noise)
        outliers = []
        outlier_labels = []  # outlier를 생성한 클래스의 레이블을 저장

        classes = np.unique(y_train)

        for _ in range(n_feature_outliers):
            # pick a class to generate outlier
            cls = rng.choice(classes)
            cls_idx = np.where(y_train == cls)[0]
            X_cls = X_train[cls_idx]

            # compute μ_y and Σ_y
            mu = X_cls.mean(axis=0)
            cov = np.cov(X_cls.T) + np.eye(X_train.shape[1]) * 1e-6
            inv_cov = np.linalg.inv(cov)

            # generate a far-away point
            while True:
                # sample from Gaussian
                x_candidate = rng.multivariate_normal(mu, cov)

                # compute Mahalanobis distance
                d = (x_candidate - mu).T @ inv_cov @ (x_candidate - mu)

                # 99th percentile threshold for Type I outlier
                tau = np.percentile([((x - mu).T @ inv_cov @ (x - mu)) for x in X_cls], 99)

                if d > tau:  # valid outlier
                    outliers.append(x_candidate)
                    outlier_labels.append(cls)  # 생성에 사용한 클래스 레이블 저장
                    break

        outliers = np.array(outliers)
        outlier_labels = np.array(outlier_labels)  # Type I outlier는 원래 클래스 레이블 유지

        # append to dataset
        X = np.vstack([X, outliers])
        y = np.concatenate([y, outlier_labels])

    return X, y
