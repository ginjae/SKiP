import numpy as np
import matplotlib.pyplot as plt

def create_decision_grid(X, margin=2, resolution=300):
    """결정 경계를 그리기 위한 그리드 생성"""
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, resolution),
        np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid

def plot_decision_boundary(clf, X_train, y_train, X_test, y_test, title, 
                          xx=None, yy=None, grid=None, figsize=None):
    """SVM 결정 경계 시각화"""
    if xx is None or yy is None or grid is None:
        xx, yy, grid = create_decision_grid(X_train)
    
    zz = clf.decision_function(grid).reshape(xx.shape)
    
    if figsize:
        plt.figure(figsize=figsize)
    
    plt.contourf(xx, yy, zz >= 0, alpha=0.2)
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='black')
    plt.contour(xx, yy, zz, levels=[-1, 1], linewidths=1.5, colors='blue', linestyles='dashed')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, marker='^', edgecolor='k', label='test')
    
    sv = clf.support_
    plt.scatter(X_train[sv, 0], X_train[sv, 1], s=80, facecolors='none', 
                edgecolors='r', label='SV (train)')
    plt.legend()
    plt.title(title)
    plt.show()
    
    return xx, yy, zz

def plot_decision_boundary_with_annotations(clf, X_train, y_train, X_test, y_test, 
                                           title, annotation_values, xx, yy, zz, figsize=(16, 10)):
    """어노테이션이 포함된 결정 경계 시각화"""
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, zz >= 0, alpha=0.2)
    plt.contour(xx, yy, zz, levels=[0], linewidths=2, colors='black')
    plt.contour(xx, yy, zz, levels=[-1, 1], linewidths=1.5, colors='blue', linestyles='dashed')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k', label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, marker='^', edgecolor='k', label='test')
    
    sv = clf.support_
    plt.scatter(X_train[sv, 0], X_train[sv, 1], s=80, facecolors='none', 
                edgecolors='r', label='SV (train)')
    
    # Support vector에 어노테이션 추가
    for idx in sv:
        p_val = annotation_values[idx]
        if y_train[idx] == 0:
            plt.annotate(f'{p_val:.3f}', 
                        xy=(X_train[idx, 0], X_train[idx, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        else:
            plt.annotate(f'{p_val:.3f}', 
                        xy=(X_train[idx, 0], X_train[idx, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.legend()
    plt.title(title)
    plt.show()