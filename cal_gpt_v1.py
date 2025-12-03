# requirements:
# pip install econml scikit-learn matplotlib pandas numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from econml.dml import NonParamDML
import warnings
warnings.filterwarnings("ignore")

RND = 42

def fit_dml_and_calibrate(X, T, Y, n_splits=3, n_bins=10, delta_frac=0.01):
    """
    X : (n_samples, n_features) numpy array or DataFrame
    T : (n_samples,) continuous treatment (1-D)
    Y : (n_samples,) binary outcome (0/1)
    Returns: dict with estimator, predicted_marginal, calibration results, and plots
    """
    # Ensure numpy arrays
    if isinstance(X, pd.DataFrame):
        X_np = X.values
    else:
        X_np = np.asarray(X)
    T_np = np.asarray(T).reshape(-1, 1)
    Y_np = np.asarray(Y).ravel()
    n = X_np.shape[0]

    # 1) Fit a NonParamDML for continuous treatment + binary outcome
    # - model_y should be a classifier if Y is binary (set discrete_outcome=True)
    # - model_t is regressor for continuous T
    # - model_final is a regressor that learns the (featurized) CATE / marginal effect
    model_y = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=RND)
    model_t = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, random_state=RND)
    model_final = RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=RND)

    dml = NonParamDML(
        model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        discrete_outcome=True,      # Y is binary -> instruct internals to use classifiers where appropriate
        discrete_treatment=False,
        cv=3,
        random_state=RND
    )

    dml.fit(Y_np, T_np, X=X_np)
    # We'll compute the marginal effect at each unit's observed T (i.e., ∂τ(T_i, X_i))
    # NonParamDML.marginal_effect expects a base-treatment array shape (n, d_t)
    predicted_marginal = dml.marginal_effect(T_np, X=X_np).ravel()  # shape (n,)

    # 2) Build an out-of-fold AIPW pseudo-outcome to act as an "observed" target for calibration.
    #    Because T is continuous, we discretize for this diagnostic: A = (T >= threshold).
    #    NOTE: this changes estimand to a binary contrast (high vs low), but is a practical diagnostic.
    threshold = np.median(T_np)              # you can choose other thresholds or multiple
    A_np = (T_np.ravel() >= threshold).astype(int)

    # Prepare arrays for cross-fitted predictions
    e_hat = np.zeros(n)      # propensity P(A=1 | X)
    m1_hat = np.zeros(n)     # E[Y | X, A=1]
    m0_hat = np.zeros(n)     # E[Y | X, A=0]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RND)
    for train_idx, val_idx in kf.split(X_np):
        # Propensity model (p(A=1|X))
        prop = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=RND)
        prop.fit(X_np[train_idx], A_np[train_idx])
        e_hat[val_idx] = prop.predict_proba(X_np[val_idx])[:, 1]

        # Outcome regressions for treated/control (out-of-fold)
        # If a group has too few samples in training set, fallback to using a single model with A as input.
        if np.sum(A_np[train_idx] == 1) >= 20 and np.sum(A_np[train_idx] == 0) >= 20:
            m1 = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=RND)
            m0 = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, random_state=RND)
            # Fit on each subset
            m1.fit(X_np[train_idx][A_np[train_idx]==1], Y_np[train_idx][A_np[train_idx]==1])
            m0.fit(X_np[train_idx][A_np[train_idx]==0], Y_np[train_idx][A_np[train_idx]==0])
            m1_hat[val_idx] = m1.predict_proba(X_np[val_idx])[:, 1]
            m0_hat[val_idx] = m0.predict_proba(X_np[val_idx])[:, 1]
        else:
            # fallback: single outcome model with A as input
            from sklearn.ensemble import RandomForestClassifier as RFC
            out_model = RFC(n_estimators=200, min_samples_leaf=10, random_state=RND)
            XA_train = np.hstack([X_np[train_idx], A_np[train_idx].reshape(-1,1)])
            out_model.fit(XA_train, Y_np[train_idx])
            XA_val1 = np.hstack([X_np[val_idx], np.ones((len(val_idx),1))])
            XA_val0 = np.hstack([X_np[val_idx], np.zeros((len(val_idx),1))])
            m1_hat[val_idx] = out_model.predict_proba(XA_val1)[:,1]
            m0_hat[val_idx] = out_model.predict_proba(XA_val0)[:,1]

    # Clip propensity to avoid division by zero instability
    eps = 1e-6
    e_hat = np.clip(e_hat, eps, 1 - eps)

    # AIPW pseudo-outcome (doubly robust):
    # phi = (A*(Y - m1_hat))/e_hat - ((1-A)*(Y - m0_hat))/(1-e_hat) + (m1_hat - m0_hat)
    phi = (A_np * (Y_np - m1_hat) / e_hat) - ((1 - A_np) * (Y_np - m0_hat) / (1 - e_hat)) + (m1_hat - m0_hat)

    # 3) Calibration diagnostics:
    # 3a) OLS of phi on predicted_marginal (slope near 1 means good calibration)
    reg = LinearRegression().fit(predicted_marginal.reshape(-1,1), phi)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(predicted_marginal.reshape(-1,1), phi)

    # 3b) Quantile (GATE) calibration: bin predicted CATE and compare group means
    df = pd.DataFrame({
        'predicted_marginal': predicted_marginal,
        'phi': phi,
        'A': A_np,
        'T': T_np.ravel()
    })
    df['q'] = pd.qcut(df['predicted_marginal'], q=n_bins, duplicates='drop')
    group = df.groupby('q').agg(
        mean_pred = ('predicted_marginal', 'mean'),
        mean_phi = ('phi', 'mean'),
        size = ('phi', 'size'),
        mean_T = ('T','mean')
    ).reset_index()

    # Plot calibration curve
    plt.figure(figsize=(7,6))
    plt.plot(group['mean_pred'], group['mean_phi'], marker='o', linestyle='-',
             label='Observed (AIPW phi) per quantile')
    minv = min(group['mean_pred'].min(), group['mean_phi'].min())
    maxv = max(group['mean_pred'].max(), group['mean_phi'].max())
    plt.plot([minv, maxv], [minv, maxv], linestyle='--', color='k', label='45° perfect calibration')
    plt.xlabel('Mean predicted marginal effect (per quantile)')
    plt.ylabel('Mean pseudo-outcome (AIPW) per quantile')
    plt.title('Quantile calibration: predicted vs observed (AIPW)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Also show scatter of individual phi vs predicted_marginal and OLS line
    plt.figure(figsize=(7,6))
    plt.scatter(predicted_marginal, phi, alpha=0.3, s=12)
    xline = np.linspace(predicted_marginal.min(), predicted_marginal.max(), 50)
    plt.plot(xline, reg.predict(xline.reshape(-1,1)), color='red', linewidth=2, label=f'OLS slope={slope:.3f}, intercept={intercept:.3f}')
    plt.plot([xline.min(), xline.max()], [xline.min(), xline.max()], linestyle='--', color='k', label='45° ideal')
    plt.xlabel('Predicted marginal effect ∂τ/∂T (per sample)')
    plt.ylabel('AIPW pseudo-outcome (binary contrast)')
    plt.title('Individual pseudo-outcome vs predicted marginal CATE')
    plt.legend()
    plt.grid(True)
    plt.show()

    results = {
        'estimator': dml,
        'predicted_marginal': predicted_marginal,
        'phi': phi,
        'calib_slope': slope,
        'calib_intercept': intercept,
        'calib_r2': r2,
        'group_table': group,
        'threshold_used': threshold
    }
    return results

# Example usage with synthetic data (toy)
if __name__ == "__main__":
    # small synthetic example (replace with your real data)
    from sklearn.datasets import make_classification
    n = 2000
    Xs, _ = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=RND)
    # create continuous treatment as a smooth function of X + noise
    T_cont = (Xs[:,0] * 0.8 + Xs[:,1] * -0.5 + np.random.normal(scale=1.0, size=n))
    # binary outcome as logistic of (treatment effect times T + covariate baseline)
    true_tau = 0.2 + 0.1 * (Xs[:,2] > 0)   # toy heterogeneous marginal effect
    baseline = -0.2 + 0.3 * Xs[:,3]
    logits = baseline + true_tau * T_cont
    probs = 1 / (1 + np.exp(-logits))
    Y_bin = (np.random.rand(n) < probs).astype(int)

    res = fit_dml_and_calibrate(Xs, T_cont, Y_bin, n_splits=3, n_bins=10)

    print("Calibration OLS slope (phi ~ pred):", res['calib_slope'])
    print("Intercept:", res['calib_intercept'])
    print("R^2:", res['calib_r2'])
    print(res['group_table'][['mean_pred','mean_phi','size']])
