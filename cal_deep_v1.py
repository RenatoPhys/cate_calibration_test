import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from econml.dml import DML, LinearDML, CausalForestDML
from econml.metalearners import TLearner, SLearner, XLearner
import warnings
warnings.filterwarnings('ignore')

class CATE_Calibrator:
    """
    Class to calibrate CATE models from EconML with continuous treatment and binary outcome
    """
    
    def __init__(self, model, calibration_method='doubly_robust', n_bins=10, random_state=42):
        """
        Initialize the calibrator
        
        Parameters:
        -----------
        model : EconML model
            Trained CATE model
        calibration_method : str
            Method for calibration: 'doubly_robust', 'matching', 'stratified'
        n_bins : int
            Number of bins for calibration
        random_state : int
            Random seed
        """
        self.model = model
        self.calibration_method = calibration_method
        self.n_bins = n_bins
        self.random_state = random_state
        self.calibration_results = {}
        
    def create_calibration_dataset(self, X, T, y, test_size=0.3):
        """
        Create calibration dataset using cross-fitting
        
        Parameters:
        -----------
        X : array-like
            Features
        T : array-like
            Continuous treatment
        y : array-like
            Binary outcome
        test_size : float
            Size of calibration set
            
        Returns:
        --------
        X_calib, T_calib, y_calib : calibration dataset
        """
        X_train, X_calib, T_train, T_calib, y_train, y_calib = train_test_split(
            X, T, y, test_size=test_size, random_state=self.random_state
        )
        
        # Re-train model on training set if needed
        # Note: If model is already trained, skip this or use cross-fitting
        return X_calib, T_calib, y_calib
    
    def predict_cate_for_contrast(self, X, T_low=None, T_high=None):
        """
        Predict CATE for specific treatment contrast
        
        Parameters:
        -----------
        X : array-like
            Features
        T_low, T_high : float
            Treatment levels for contrast. If None, use unit change
        """
        if T_low is None or T_high is None:
            # For DML models, effect method gives constant marginal effect
            try:
                # For LinearDML and similar
                cate = self.model.effect(X)
            except:
                # For other models, approximate with unit change
                T0 = np.zeros((len(X), 1))
                T1 = np.ones((len(X), 1))
                cate = self.model.effect(X, T0=T0, T1=T1)
        else:
            # Specific contrast
            T0 = np.full((len(X), 1), T_low)
            T1 = np.full((len(X), 1), T_high)
            cate = self.model.effect(X, T0=T0, T1=T1)
            
        return cate
    
    def doubly_robust_scores(self, X, T, y, n_splits=5):
        """
        Compute doubly robust scores for calibration
        
        Parameters:
        -----------
        X : array-like
            Features
        T : array-like
            Treatment
        y : array-like
            Outcome
        n_splits : int
            Number of cross-fitting splits
            
        Returns:
        --------
        dr_scores : array
            Doubly robust scores for each observation
        """
        n_samples = len(X)
        dr_scores = np.zeros(n_samples)
        
        # Use cross-fitting to avoid overfitting
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            T_train, T_val = T[train_idx], T[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit outcome model (Y ~ X, T)
            # For continuous treatment, we can use different approaches:
            
            # Method 1: Polynomial expansion for treatment
            XT_train = np.column_stack([X_train, T_train, T_train**2])
            XT_val = np.column_stack([X_val, T_val, T_val**2])
            
            outcome_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state
            )
            outcome_model.fit(XT_train, y_train)
            
            # Fit propensity model (T ~ X)
            # For continuous treatment, use regression
            propensity_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state
            )
            propensity_model.fit(X_train, T_train)
            
            # Predict propensity scores
            T_pred = propensity_model.predict(X_val)
            propensity_errors = T_val - T_pred
            propensity_var = np.var(propensity_errors)
            
            # Predict outcomes
            y_pred = outcome_model.predict(XT_val)
            
            # Create counterfactual predictions
            # For unit change in treatment
            T_counterfactual = T_val + 1
            XT_counterfactual = np.column_stack([X_val, T_counterfactual, T_counterfactual**2])
            y_pred_cf = outcome_model.predict(XT_counterfactual)
            
            # Compute doubly robust scores
            # Formula adapted for continuous treatment
            ipw_weights = 1 / (np.sqrt(2 * np.pi * propensity_var) * 
                             np.exp(-0.5 * (T_val - T_pred)**2 / propensity_var))
            
            # Stabilize weights
            ipw_weights = np.clip(ipw_weights, np.percentile(ipw_weights, 1), 
                                np.percentile(ipw_weights, 99))
            
            dr_scores[val_idx] = (y_val - y_pred) * (T_val - T_pred) / propensity_var + (y_pred_cf - y_pred)
            
        return dr_scores
    
    def calibrate(self, X, T, y, contrast=None, n_splits=5):
        """
        Main calibration method
        
        Parameters:
        -----------
        X : array-like
            Features
        T : array-like
            Treatment
        y : array-like
            Outcome
        contrast : tuple
            (T_low, T_high) for specific contrast
        n_splits : int
            Number of cross-fitting splits
        """
        # Predict CATE
        if contrast:
            T_low, T_high = contrast
            cate_pred = self.predict_cate_for_contrast(X, T_low, T_high)
        else:
            cate_pred = self.predict_cate_for_contrast(X)
        
        if self.calibration_method == 'doubly_robust':
            actual_effects = self.doubly_robust_scores(X, T, y, n_splits)
        elif self.calibration_method == 'matching':
            actual_effects = self.matching_estimator(X, T, y)
        elif self.calibration_method == 'stratified':
            actual_effects = self.stratified_estimator(X, T, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        # Group by predicted CATE
        bins = np.percentile(cate_pred, np.linspace(0, 100, self.n_bins + 1))
        bin_indices = np.digitize(cate_pred, bins)
        
        calibration_data = []
        for i in range(1, self.n_bins + 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                avg_pred = cate_pred[mask].mean()
                avg_actual = actual_effects[mask].mean()
                std_actual = actual_effects[mask].std()
                n_samples = mask.sum()
                
                calibration_data.append({
                    'bin': i,
                    'predicted_cate': avg_pred,
                    'actual_effect': avg_actual,
                    'std_error': std_actual / np.sqrt(n_samples),
                    'n_samples': n_samples
                })
        
        self.calibration_results = pd.DataFrame(calibration_data)
        return self.calibration_results
    
    def plot_calibration(self, figsize=(10, 6)):
        """
        Plot calibration curve
        """
        if self.calibration_results.empty:
            raise ValueError("Run calibrate() first")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Calibration plot
        ax = axes[0]
        results = self.calibration_results
        ax.errorbar(results['predicted_cate'], results['actual_effect'],
                   yerr=1.96*results['std_error'], fmt='o', capsize=5)
        ax.plot([results['predicted_cate'].min(), results['predicted_cate'].max()],
               [results['predicted_cate'].min(), results['predicted_cate'].max()], 
               'r--', label='Perfect calibration')
        ax.set_xlabel('Predicted CATE')
        ax.set_ylabel('Estimated Actual Effect')
        ax.set_title('CATE Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Distribution of predictions vs actual
        ax = axes[1]
        width = 0.35
        x = np.arange(len(results))
        ax.bar(x - width/2, results['predicted_cate'], width, label='Predicted', alpha=0.7)
        ax.bar(x + width/2, results['actual_effect'], width, label='Actual', alpha=0.7)
        ax.set_xlabel('Bin')
        ax.set_ylabel('Effect Size')
        ax.set_title('Predicted vs Actual Effects by Bin')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def calibration_metrics(self):
        """
        Calculate calibration metrics
        """
        if self.calibration_results.empty:
            raise ValueError("Run calibrate() first")
        
        results = self.calibration_results
        
        # Mean Absolute Calibration Error
        mace = np.mean(np.abs(results['predicted_cate'] - results['actual_effect']))
        
        # Root Mean Square Calibration Error
        rmsce = np.sqrt(np.mean((results['predicted_cate'] - results['actual_effect'])**2))
        
        # Correlation between predicted and actual
        correlation = np.corrcoef(results['predicted_cate'], results['actual_effect'])[0, 1]
        
        # Calibration slope (linear regression of actual on predicted)
        if len(results) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                results['predicted_cate'], results['actual_effect']
            )
        else:
            slope, intercept, r_value = np.nan, np.nan, np.nan
        
        metrics = {
            'mean_absolute_calibration_error': mace,
            'root_mean_square_calibration_error': rmsce,
            'correlation': correlation,
            'calibration_slope': slope,
            'calibration_intercept': intercept,
            'r_squared': r_value**2 if not np.isnan(r_value) else np.nan
        }
        
        return metrics
    
    def bootstrap_calibration(self, X, T, y, n_bootstraps=100, contrast=None):
        """
        Bootstrap calibration to get confidence intervals
        """
        n_samples = len(X)
        bootstrap_metrics = []
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            T_boot = T[indices]
            y_boot = y[indices]
            
            # Calibrate on bootstrap sample
            self.calibrate(X_boot, T_boot, y_boot, contrast)
            metrics = self.calibration_metrics()
            bootstrap_metrics.append(metrics)
        
        # Calculate confidence intervals
        bootstrap_df = pd.DataFrame(bootstrap_metrics)
        ci_lower = bootstrap_df.quantile(0.025)
        ci_upper = bootstrap_df.quantile(0.975)
        mean_metrics = bootstrap_df.mean()
        
        ci_results = pd.DataFrame({
            'mean': mean_metrics,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })
        
        return ci_results

# Example usage function
def example_usage():
    """
    Example of how to use the CATE calibrator
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 2000
    
    # Features
    X = np.random.randn(n_samples, 5)
    
    # Continuous treatment with heterogeneity
    T = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n_samples) * 0.5
    
    # True CATE (heterogeneous treatment effect)
    true_cate = 0.8 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    
    # Binary outcome with continuous treatment effect
    treatment_effect = true_cate * T
    log_odds = 0.3 * X[:, 0] + 0.2 * X[:, 1] + treatment_effect - 0.5
    prob = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, prob)
    
    # Split data
    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(
        X, T, y, test_size=0.3, random_state=42
    )
    
    # Train DML model from EconML
    print("Training DML model...")
    model = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, random_state=42),
        discrete_treatment=False,
        linear_first_stages=False,
        random_state=42
    )
    
    # Reshape T for EconML (needs 2D for continuous treatment)
    T_train_2d = T_train.reshape(-1, 1)
    T_test_2d = T_test.reshape(-1, 1)
    
    model.fit(y_train, T_train_2d, X=X_train)
    
    # Initialize calibrator
    calibrator = CATE_Calibrator(
        model=model,
        calibration_method='doubly_robust',
        n_bins=10,
        random_state=42
    )
    
    # Calibrate on test set
    print("Calibrating model...")
    calibration_results = calibrator.calibrate(X_test, T_test, y_test)
    
    # Plot calibration
    print("Generating calibration plot...")
    fig = calibrator.plot_calibration()
    plt.show()
    
    # Calculate metrics
    metrics = calibrator.calibration_metrics()
    print("\nCalibration Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Bootstrap confidence intervals
    print("\nCalculating bootstrap confidence intervals...")
    ci_results = calibrator.bootstrap_calibration(X_test, T_test, y_test, n_bootstraps=50)
    print("\nBootstrap Results:")
    print(ci_results)
    
    return calibrator, metrics, ci_results

def compare_calibration_methods(X, T, y, models_dict, n_bins=10):
    """
    Compare calibration across different CATE models
    """
    results = {}
    
    for name, model in models_dict.items():
        print(f"\nCalibrating {name}...")
        calibrator = CATE_Calibrator(
            model=model,
            calibration_method='doubly_robust',
            n_bins=n_bins
        )
        
        calibrator.calibrate(X, T, y)
        metrics = calibrator.calibration_metrics()
        results[name] = metrics
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('mean_absolute_calibration_error')
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ['mean_absolute_calibration_error', 'root_mean_square_calibration_error', 'correlation']
    
    x = np.arange(len(comparison_df))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 1) * width
        ax.bar(x + offset, comparison_df[metric], width, label=metric)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Calibration Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# Advanced calibration with dose-response curves
def dose_response_calibration(model, X, T, y, treatment_grid=None, n_groups=5):
    """
    Calibrate dose-response predictions
    """
    if treatment_grid is None:
        treatment_grid = np.linspace(np.percentile(T, 10), np.percentile(T, 90), 20)
    
    # Group by predicted CATE magnitude
    cate_pred = model.effect(X)
    cate_bins = np.percentile(cate_pred, np.linspace(0, 100, n_groups + 1))
    group_indices = np.digitize(cate_pred, cate_bins)
    
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 5))
    
    for group in range(1, n_groups + 1):
        mask = group_indices == group
        if mask.sum() == 0:
            continue
        
        X_group = X[mask]
        T_group = T[mask]
        y_group = y[mask]
        
        # Predict dose-response for each observation
        dose_responses = []
        for t in treatment_grid:
            T0 = np.full((len(X_group), 1), treatment_grid[0])
            T1 = np.full((len(X_group), 1), t)
            effect = model.effect(X_group, T0=T0, T1=T1)
            dose_responses.append(effect.mean())
        
        ax = axes[group-1] if n_groups > 1 else axes
        ax.plot(treatment_grid, dose_responses, 'b-', label='Predicted', linewidth=2)
        
        # Estimate actual dose-response using local linear regression
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y_group, T_group, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'r--', label='Actual (smoothed)', linewidth=2)
        
        ax.set_xlabel('Treatment Level')
        ax.set_ylabel('Outcome Probability')
        ax.set_title(f'CATE Group {group} (n={mask.sum()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Run example
    calibrator, metrics, ci_results = example_usage()