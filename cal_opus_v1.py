import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.isotonic import IsotonicRegression
from econml.dml import LinearDML, CausalForestDML, NonParamDML
from econml.dr import DRLearner
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# 1. GENERATE SYNTHETIC DATA (continuous T, binary Y)
# =============================================================================
def generate_data(n=10000, seed=42):
    """
    Generate synthetic data with known true CATE for validation.
    Y = Bernoulli(sigmoid(alpha + tau(X)*T + X@beta + noise))
    """
    np.random.seed(seed)
    
    # Confounders
    X = np.random.randn(n, 5)
    
    # True CATE function (heterogeneous treatment effect)
    true_cate = 0.5 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 0] * X[:, 1]
    
    # Treatment (continuous, depends on confounders)
    propensity = 0.5 + 0.2 * X[:, 0] + 0.1 * X[:, 2]
    T = propensity + 0.5 * np.random.randn(n)
    
    # Outcome model (binary)
    baseline = -1 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    logit = baseline + true_cate * T
    prob = 1 / (1 + np.exp(-logit))
    Y = np.random.binomial(1, prob)
    
    return X, T, Y, true_cate


# =============================================================================
# 2. CALIBRATED OUTCOME MODEL WRAPPER
# =============================================================================
class CalibratedBinaryOutcomeModel:
    """
    Wrapper to ensure model_y outputs calibrated probabilities.
    Critical for binary outcomes in DML.
    """
    def __init__(self, base_estimator=None, method='isotonic', cv=5):
        if base_estimator is None:
            base_estimator = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, random_state=42
            )
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.calibrated_model = None
    
    def fit(self, X, y):
        self.calibrated_model = CalibratedClassifierCV(
            self.base_estimator, 
            method=self.method, 
            cv=self.cv
        )
        self.calibrated_model.fit(X, y)
        return self
    
    def predict(self, X):
        # DML expects predict() to return E[Y|X], so return probabilities
        return self.calibrated_model.predict_proba(X)[:, 1]
    
    def predict_proba(self, X):
        return self.calibrated_model.predict_proba(X)


# =============================================================================
# 3. CATE CALIBRATION CLASS
# =============================================================================
class CATECalibrator:
    """
    Calibrate CATE estimates using isotonic regression or binning.
    
    The key insight: we can't observe individual treatment effects,
    but we can check calibration by:
    1. Binning by predicted CATE
    2. Within each bin, estimating average effect via regression adjustment
    3. Comparing predicted vs. realized effects
    """
    
    def __init__(self, method='isotonic', n_bins=10):
        self.method = method
        self.n_bins = n_bins
        self.calibrator = None
        self.bin_edges = None
        self.bin_means_pred = None
        self.bin_means_actual = None
        
    def fit(self, cate_pred, X, T, Y, model_y_residual=None):
        """
        Fit calibrator using a validation set.
        
        For continuous treatment, we estimate actual CATE in bins using
        the relationship: E[Y|T,X] ≈ E[Y|X] + CATE(X) * (T - E[T|X])
        
        Parameters:
        -----------
        cate_pred : predicted CATE values
        X : confounders
        T : treatment (continuous)
        Y : outcome (binary)
        model_y_residual : residuals from outcome model (optional)
        """
        
        if self.method == 'isotonic':
            # For isotonic, we need actual effects - estimate via local regression
            actual_effects = self._estimate_actual_effects_continuous(
                cate_pred, X, T, Y
            )
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(cate_pred, actual_effects)
            
        elif self.method == 'binning':
            # Bin by predicted CATE and compute actual effect in each bin
            self.bin_edges = np.percentile(
                cate_pred, 
                np.linspace(0, 100, self.n_bins + 1)
            )
            bin_indices = np.digitize(cate_pred, self.bin_edges[1:-1])
            
            self.bin_means_pred = []
            self.bin_means_actual = []
            
            for b in range(self.n_bins):
                mask = bin_indices == b
                if mask.sum() < 50:  # Need enough samples
                    continue
                    
                self.bin_means_pred.append(cate_pred[mask].mean())
                
                # Estimate actual effect in this bin using linear regression
                actual = self._estimate_bin_effect(T[mask], Y[mask], X[mask])
                self.bin_means_actual.append(actual)
            
            self.bin_means_pred = np.array(self.bin_means_pred)
            self.bin_means_actual = np.array(self.bin_means_actual)
            
            # Fit linear calibration
            if len(self.bin_means_pred) > 2:
                slope, intercept, _, _, _ = stats.linregress(
                    self.bin_means_pred, self.bin_means_actual
                )
                self.calibrator = lambda x: slope * x + intercept
        
        return self
    
    def _estimate_actual_effects_continuous(self, cate_pred, X, T, Y):
        """
        Estimate actual treatment effects for continuous treatment.
        Uses a local linear regression approach within CATE quantile bins.
        """
        n_calib_bins = 20
        bin_edges = np.percentile(cate_pred, np.linspace(0, 100, n_calib_bins + 1))
        bin_indices = np.digitize(cate_pred, bin_edges[1:-1])
        
        actual_effects = np.zeros_like(cate_pred)
        
        for b in range(n_calib_bins):
            mask = bin_indices == b
            if mask.sum() < 30:
                actual_effects[mask] = cate_pred[mask]
                continue
            
            # Within-bin regression: Y ~ T + X to get coefficient on T
            X_bin = np.column_stack([T[mask], X[mask]])
            Y_bin = Y[mask]
            
            try:
                # Use robust regression
                from sklearn.linear_model import Ridge
                reg = Ridge(alpha=1.0)
                reg.fit(X_bin, Y_bin)
                actual_effects[mask] = reg.coef_[0]  # Coefficient on T
            except:
                actual_effects[mask] = cate_pred[mask]
        
        return actual_effects
    
    def _estimate_bin_effect(self, T, Y, X):
        """Estimate average treatment effect within a bin."""
        from sklearn.linear_model import LogisticRegression
        
        # Partial out X and estimate effect of T on Y
        X_with_T = np.column_stack([T.reshape(-1, 1), X])
        
        try:
            # For binary outcome, use logistic regression and get marginal effect
            model = LogisticRegression(max_iter=1000)
            model.fit(X_with_T, Y)
            # Marginal effect ≈ coefficient * mean(p*(1-p))
            probs = model.predict_proba(X_with_T)[:, 1]
            marginal_effect = model.coef_[0, 0] * np.mean(probs * (1 - probs))
            return marginal_effect
        except:
            # Fallback to simple regression
            return np.corrcoef(T, Y)[0, 1] * np.std(Y) / np.std(T)
    
    def transform(self, cate_pred):
        """Apply calibration to new predictions."""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'isotonic':
            return self.calibrator.transform(cate_pred)
        else:
            return self.calibrator(cate_pred)
    
    def plot_calibration(self, cate_pred, X, T, Y, true_cate=None, ax=None):
        """Plot calibration curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute binned actual effects
        n_plot_bins = 10
        bin_edges = np.percentile(cate_pred, np.linspace(0, 100, n_plot_bins + 1))
        bin_indices = np.digitize(cate_pred, bin_edges[1:-1])
        
        pred_means = []
        actual_means = []
        true_means = [] if true_cate is not None else None
        
        for b in range(n_plot_bins):
            mask = bin_indices == b
            if mask.sum() < 30:
                continue
            pred_means.append(cate_pred[mask].mean())
            actual_means.append(self._estimate_bin_effect(T[mask], Y[mask], X[mask]))
            if true_cate is not None:
                true_means.append(true_cate[mask].mean())
        
        pred_means = np.array(pred_means)
        actual_means = np.array(actual_means)
        
        # Plot
        ax.scatter(pred_means, actual_means, s=100, label='Estimated actual effect', zorder=3)
        
        if true_cate is not None:
            true_means = np.array(true_means)
            ax.scatter(pred_means, true_means, s=100, marker='x', 
                      label='True CATE (if known)', zorder=3)
        
        # Perfect calibration line
        lims = [min(pred_means.min(), actual_means.min()) - 0.1,
                max(pred_means.max(), actual_means.max()) + 0.1]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')
        
        ax.set_xlabel('Predicted CATE')
        ax.set_ylabel('Actual CATE (estimated)')
        ax.set_title('CATE Calibration Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax


# =============================================================================
# 4. FULL PIPELINE: DML WITH CALIBRATION
# =============================================================================
def fit_calibrated_dml(X, T, Y, X_val, T_val, Y_val, 
                       dml_type='forest', calibrate_cate=True):
    """
    Fit DML with calibrated outcome model and optional CATE calibration.
    """
    
    # Step 1: Create calibrated outcome model
    calibrated_model_y = CalibratedBinaryOutcomeModel(
        base_estimator=GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
        method='isotonic',
        cv=5
    )
    
    # Step 2: Treatment model (continuous T, so regression)
    model_t = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, random_state=42
    )
    
    # Step 3: Fit DML
    if dml_type == 'linear':
        dml = LinearDML(
            model_y=calibrated_model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=5,
            random_state=42
        )
    elif dml_type == 'forest':
        dml = CausalForestDML(
            model_y=calibrated_model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=5,
            n_estimators=200,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown DML type: {dml_type}")
    
    print("Fitting DML model...")
    dml.fit(Y, T, X=X)
    
    # Step 4: Get CATE predictions
    cate_train = dml.effect(X)
    cate_val = dml.effect(X_val)
    
    # Step 5: Calibrate CATE (optional but recommended)
    cate_calibrator = None
    if calibrate_cate:
        print("Calibrating CATE estimates...")
        cate_calibrator = CATECalibrator(method='isotonic', n_bins=10)
        cate_calibrator.fit(cate_val, X_val, T_val, Y_val)
        cate_val_calibrated = cate_calibrator.transform(cate_val)
    else:
        cate_val_calibrated = cate_val
    
    return dml, cate_calibrator, cate_val, cate_val_calibrated


# =============================================================================
# 5. EVALUATION METRICS
# =============================================================================
def evaluate_cate_calibration(cate_pred, true_cate, X, T, Y):
    """Compute calibration metrics for CATE estimates."""
    
    metrics = {}
    
    # 1. If true CATE known: RMSE and correlation
    if true_cate is not None:
        metrics['rmse_vs_true'] = np.sqrt(np.mean((cate_pred - true_cate)**2))
        metrics['corr_vs_true'] = np.corrcoef(cate_pred, true_cate)[0, 1]
        metrics['bias'] = np.mean(cate_pred - true_cate)
    
    # 2. Calibration slope (should be ~1 for well-calibrated)
    calibrator = CATECalibrator(method='binning', n_bins=10)
    calibrator.fit(cate_pred, X, T, Y)
    
    if len(calibrator.bin_means_pred) > 2:
        slope, intercept, r, p, se = stats.linregress(
            calibrator.bin_means_pred, 
            calibrator.bin_means_actual
        )
        metrics['calibration_slope'] = slope
        metrics['calibration_intercept'] = intercept
        metrics['calibration_r2'] = r**2
    
    # 3. Mean calibration error (like ECE for probabilities)
    if calibrator.bin_means_pred is not None and len(calibrator.bin_means_pred) > 0:
        metrics['mean_calibration_error'] = np.mean(
            np.abs(calibrator.bin_means_pred - calibrator.bin_means_actual)
        )
    
    return metrics


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Generate data
    print("Generating synthetic data...")
    X, T, Y, true_cate = generate_data(n=10000, seed=42)
    
    # Train/validation split
    (X_train, X_val, T_train, T_val, 
     Y_train, Y_val, true_cate_train, true_cate_val) = train_test_split(
        X, T, Y, true_cate, test_size=0.3, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Outcome prevalence: {Y.mean():.3f}")
    print(f"Treatment range: [{T.min():.2f}, {T.max():.2f}]")
    
    # Fit calibrated DML
    dml, cate_calibrator, cate_val_raw, cate_val_calibrated = fit_calibrated_dml(
        X_train, T_train, Y_train,
        X_val, T_val, Y_val,
        dml_type='forest',
        calibrate_cate=True
    )
    
    # Evaluate
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Raw CATE metrics
    print("\n--- Raw CATE (before calibration) ---")
    metrics_raw = evaluate_cate_calibration(
        cate_val_raw, true_cate_val, X_val, T_val, Y_val
    )
    for k, v in metrics_raw.items():
        print(f"  {k}: {v:.4f}")
    
    # Calibrated CATE metrics
    print("\n--- Calibrated CATE ---")
    metrics_cal = evaluate_cate_calibration(
        cate_val_calibrated, true_cate_val, X_val, T_val, Y_val
    )
    for k, v in metrics_cal.items():
        print(f"  {k}: {v:.4f}")
    
    # Plot calibration curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw CATE calibration
    calibrator_plot = CATECalibrator(method='binning', n_bins=10)
    calibrator_plot.fit(cate_val_raw, X_val, T_val, Y_val)
    calibrator_plot.plot_calibration(
        cate_val_raw, X_val, T_val, Y_val, 
        true_cate=true_cate_val, ax=axes[0]
    )
    axes[0].set_title('Before CATE Calibration')
    
    # Calibrated CATE
    calibrator_plot2 = CATECalibrator(method='binning', n_bins=10)
    calibrator_plot2.fit(cate_val_calibrated, X_val, T_val, Y_val)
    calibrator_plot2.plot_calibration(
        cate_val_calibrated, X_val, T_val, Y_val,
        true_cate=true_cate_val, ax=axes[1]
    )
    axes[1].set_title('After CATE Calibration')
    
    plt.tight_layout()
    plt.savefig('cate_calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nCalibration plot saved to 'cate_calibration_comparison.png'")