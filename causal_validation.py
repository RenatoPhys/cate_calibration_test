"""
Unified Causal Model Validation Module

This module provides comprehensive validation tools for causal/uplift models,
combining AUC-based metrics and curve-based visualizations inspired by
nubank/fklearn's causal validation modules.

Features:
- Area Under Uplift Curve (AUUC) computation
- Qini coefficient calculation
- Cumulative gain curves
- Uplift curves with confidence intervals
- Model comparison utilities
- Support for BOTH binary and continuous treatments
- Dose-response curve validation for continuous treatments
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import spearmanr, kendalltau
from sklearn.utils import resample


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class UpliftCurveResult:
    """Container for uplift curve computation results (binary treatment)."""
    percentiles: np.ndarray
    uplift: np.ndarray
    cumulative_gain: np.ndarray
    random_baseline: np.ndarray
    n_treatment: np.ndarray
    n_control: np.ndarray


@dataclass
class DoseResponseResult:
    """Container for dose-response curve results (continuous treatment)."""
    treatment_bins: np.ndarray
    bin_centers: np.ndarray
    observed_response: np.ndarray
    predicted_effect: np.ndarray
    n_samples: np.ndarray
    std_error: np.ndarray


@dataclass 
class ContinuousTreatmentMetrics:
    """Metrics for continuous treatment models."""
    dose_response_correlation: float  # Correlation between predicted and observed dose-response
    rank_correlation: float  # Spearman correlation
    kendall_tau: float  # Kendall's tau
    mse: float  # Mean squared error of predicted effects
    mae: float  # Mean absolute error
    calibration_slope: float  # Slope of predicted vs observed regression
    calibration_intercept: float  # Intercept of predicted vs observed


@dataclass
class AUCResult:
    """Container for AUC metric results (binary treatment)."""
    auuc: float
    auuc_normalized: float
    qini: float
    qini_normalized: float
    random_auuc: float


@dataclass
class ValidationResult:
    """Complete validation result combining curves and metrics."""
    curve_result: Union[UpliftCurveResult, DoseResponseResult]
    auc_result: Optional[AUCResult] = None  # Only for binary treatment
    continuous_metrics: Optional[ContinuousTreatmentMetrics] = None  # Only for continuous
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    treatment_type: str = "binary"  # "binary" or "continuous"


# =============================================================================
# Core Utility Functions
# =============================================================================

def _detect_treatment_type(treatment: np.ndarray) -> str:
    """
    Automatically detect if treatment is binary or continuous.
    
    Parameters
    ----------
    treatment : np.ndarray
        Treatment values
        
    Returns
    -------
    str
        "binary" or "continuous"
    """
    unique_values = np.unique(treatment[~np.isnan(treatment)])
    
    if len(unique_values) == 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
        return "binary"
    elif len(unique_values) <= 2:
        # Could be binary with different coding
        return "binary"
    else:
        return "continuous"


def _validate_inputs(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    treatment_type: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Validate and convert inputs to numpy arrays.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions (uplift scores or CATE estimates)
    treatment : array-like
        Treatment indicator (binary) or treatment dose (continuous)
    outcome : array-like
        Observed outcome variable
    weights : array-like, optional
        Sample weights
    treatment_type : str, optional
        "binary" or "continuous". If None, auto-detected.
        
    Returns
    -------
    Tuple of validated numpy arrays and treatment type string
    """
    prediction = np.asarray(prediction).ravel()
    treatment = np.asarray(treatment).ravel()
    outcome = np.asarray(outcome).ravel()
    
    n = len(prediction)
    if len(treatment) != n or len(outcome) != n:
        raise ValueError("All input arrays must have the same length")
    
    # Auto-detect or validate treatment type
    if treatment_type is None:
        treatment_type = _detect_treatment_type(treatment)
    
    if treatment_type == "binary":
        if not np.all(np.isin(treatment, [0, 1])):
            raise ValueError(
                "For binary treatment, values must be 0 or 1. "
                "Use treatment_type='continuous' for continuous treatment."
            )
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights).ravel()
        if len(weights) != n:
            raise ValueError("Weights must have the same length as other inputs")
    
    return prediction, treatment, outcome, weights, treatment_type


def _validate_inputs_binary(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate inputs for binary treatment (backward compatible).
    """
    pred, treat, out, w, ttype = _validate_inputs(
        prediction, treatment, outcome, weights, treatment_type="binary"
    )
    return pred, treat, out, w


def _compute_group_stats(
    outcome: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray
) -> Tuple[float, float, int, int]:
    """
    Compute weighted outcome rates for treatment and control groups.
    
    Parameters
    ----------
    outcome : np.ndarray
        Observed outcomes
    treatment : np.ndarray
        Treatment indicators
    weights : np.ndarray
        Sample weights
    mask : np.ndarray
        Boolean mask for subset selection
        
    Returns
    -------
    Tuple of (treatment_rate, control_rate, n_treatment, n_control)
    """
    treated_mask = mask & (treatment == 1)
    control_mask = mask & (treatment == 0)
    
    n_treatment = np.sum(treated_mask)
    n_control = np.sum(control_mask)
    
    if n_treatment == 0 or n_control == 0:
        return np.nan, np.nan, n_treatment, n_control
    
    treatment_rate = np.average(
        outcome[treated_mask], 
        weights=weights[treated_mask]
    )
    control_rate = np.average(
        outcome[control_mask], 
        weights=weights[control_mask]
    )
    
    return treatment_rate, control_rate, n_treatment, n_control


# =============================================================================
# Continuous Treatment Validation Functions
# =============================================================================

def compute_dose_response_curve(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 10,
    method: str = "quantile"
) -> DoseResponseResult:
    """
    Compute dose-response curve for continuous treatment validation.
    
    This function bins the continuous treatment and computes the observed
    response rate in each bin, compared to model predictions.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions of treatment effect (CATE estimates)
    treatment : array-like
        Continuous treatment values (dosage)
    outcome : array-like
        Observed binary outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=10
        Number of treatment bins
    method : str, default="quantile"
        Binning method: "quantile" for equal-frequency, "uniform" for equal-width
        
    Returns
    -------
    DoseResponseResult
        Container with dose-response curve data
    """
    prediction, treatment, outcome, weights, _ = _validate_inputs(
        prediction, treatment, outcome, weights, treatment_type="continuous"
    )
    
    # Create treatment bins
    if method == "quantile":
        bin_edges = np.percentile(treatment, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Handle duplicate edges
    else:  # uniform
        bin_edges = np.linspace(treatment.min(), treatment.max(), n_bins + 1)
    
    actual_n_bins = len(bin_edges) - 1
    bin_centers = np.zeros(actual_n_bins)
    observed_response = np.zeros(actual_n_bins)
    predicted_effect = np.zeros(actual_n_bins)
    n_samples = np.zeros(actual_n_bins, dtype=int)
    std_error = np.zeros(actual_n_bins)
    
    for i in range(actual_n_bins):
        if i < actual_n_bins - 1:
            mask = (treatment >= bin_edges[i]) & (treatment < bin_edges[i + 1])
        else:
            mask = (treatment >= bin_edges[i]) & (treatment <= bin_edges[i + 1])
        
        n_samples[i] = np.sum(mask)
        
        if n_samples[i] > 0:
            bin_centers[i] = np.average(treatment[mask], weights=weights[mask])
            observed_response[i] = np.average(outcome[mask], weights=weights[mask])
            predicted_effect[i] = np.average(prediction[mask], weights=weights[mask])
            
            # Standard error of the mean
            if n_samples[i] > 1:
                std_error[i] = np.std(outcome[mask]) / np.sqrt(n_samples[i])
        else:
            bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            observed_response[i] = np.nan
            predicted_effect[i] = np.nan
            std_error[i] = np.nan
    
    return DoseResponseResult(
        treatment_bins=bin_edges,
        bin_centers=bin_centers,
        observed_response=observed_response,
        predicted_effect=predicted_effect,
        n_samples=n_samples,
        std_error=std_error
    )


def compute_continuous_treatment_metrics(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> ContinuousTreatmentMetrics:
    """
    Compute validation metrics for continuous treatment effect models.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions of treatment effect (CATE)
    treatment : array-like
        Continuous treatment values
    outcome : array-like
        Observed binary outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=10
        Number of bins for dose-response computation
        
    Returns
    -------
    ContinuousTreatmentMetrics
        Container with various validation metrics
    """
    prediction, treatment, outcome, weights, _ = _validate_inputs(
        prediction, treatment, outcome, weights, treatment_type="continuous"
    )
    
    # Compute dose-response curve
    dr_result = compute_dose_response_curve(
        prediction, treatment, outcome, weights, n_bins
    )
    
    # Filter out NaN values
    valid_mask = ~np.isnan(dr_result.observed_response) & ~np.isnan(dr_result.predicted_effect)
    obs = dr_result.observed_response[valid_mask]
    pred = dr_result.predicted_effect[valid_mask]
    
    if len(obs) < 2:
        return ContinuousTreatmentMetrics(
            dose_response_correlation=np.nan,
            rank_correlation=np.nan,
            kendall_tau=np.nan,
            mse=np.nan,
            mae=np.nan,
            calibration_slope=np.nan,
            calibration_intercept=np.nan
        )
    
    # Pearson correlation
    dose_response_corr = np.corrcoef(obs, pred)[0, 1]
    
    # Rank correlations
    spearman_corr, _ = spearmanr(obs, pred)
    kendall, _ = kendalltau(obs, pred)
    
    # Error metrics
    mse = np.mean((obs - pred) ** 2)
    mae = np.mean(np.abs(obs - pred))
    
    # Calibration (linear regression of observed on predicted)
    if len(pred) > 1 and np.std(pred) > 0:
        slope, intercept = np.polyfit(pred, obs, 1)
    else:
        slope, intercept = np.nan, np.nan
    
    return ContinuousTreatmentMetrics(
        dose_response_correlation=dose_response_corr,
        rank_correlation=spearman_corr,
        kendall_tau=kendall,
        mse=mse,
        mae=mae,
        calibration_slope=slope,
        calibration_intercept=intercept
    )


def compute_treatment_effect_by_quantile(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Compute treatment effects within quantiles of predicted effect.
    
    For continuous treatment, this bins observations by predicted CATE
    and computes the relationship between treatment and outcome in each bin.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions of treatment effect
    treatment : array-like
        Continuous treatment values
    outcome : array-like
        Observed outcome
    propensity : array-like, optional
        Propensity scores for IPW adjustment
    weights : array-like, optional
        Sample weights
    n_quantiles : int, default=5
        Number of quantiles to create
        
    Returns
    -------
    pd.DataFrame
        Treatment effect statistics by prediction quantile
    """
    prediction, treatment, outcome, weights, _ = _validate_inputs(
        prediction, treatment, outcome, weights, treatment_type="continuous"
    )
    
    # Create quantiles based on predictions
    try:
        quantiles = pd.qcut(prediction, n_quantiles, labels=False, duplicates='drop')
    except ValueError:
        # If we can't create the requested number of quantiles, use fewer
        quantiles = pd.qcut(prediction, min(n_quantiles, len(np.unique(prediction))), 
                          labels=False, duplicates='drop')
    
    results = []
    for q in sorted(np.unique(quantiles)):
        mask = quantiles == q
        
        q_treatment = treatment[mask]
        q_outcome = outcome[mask]
        q_weights = weights[mask]
        q_pred = prediction[mask]
        
        # Compute correlation between treatment and outcome in this quantile
        if len(q_treatment) > 2:
            treat_outcome_corr = np.corrcoef(q_treatment, q_outcome)[0, 1]
            
            # Simple linear effect estimate
            if np.std(q_treatment) > 0:
                slope, _ = np.polyfit(q_treatment, q_outcome, 1, w=q_weights)
            else:
                slope = np.nan
        else:
            treat_outcome_corr = np.nan
            slope = np.nan
        
        results.append({
            'quantile': q + 1,
            'n_samples': np.sum(mask),
            'mean_predicted_effect': np.average(q_pred, weights=q_weights),
            'mean_treatment': np.average(q_treatment, weights=q_weights),
            'mean_outcome': np.average(q_outcome, weights=q_weights),
            'treatment_outcome_corr': treat_outcome_corr,
            'estimated_slope': slope
        })
    
    return pd.DataFrame(results)


def compute_grf_rank_weighted_average(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute Rank-Weighted Average Treatment Effect (RATE) metric.
    
    This metric evaluates whether observations with higher predicted effects
    actually have higher observed effects, commonly used in GRF evaluation.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions of treatment effect
    treatment : array-like
        Treatment values (continuous)
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
        
    Returns
    -------
    float
        RATE metric value
    """
    prediction, treatment, outcome, weights, _ = _validate_inputs(
        prediction, treatment, outcome, weights, treatment_type="continuous"
    )
    
    # Rank predictions
    n = len(prediction)
    ranks = np.argsort(np.argsort(prediction)) / n  # Normalize to [0, 1]
    
    # Weight by rank (higher predicted effect = higher weight)
    rank_weights = ranks * weights
    
    # Compute weighted correlation
    weighted_outcome = outcome * rank_weights
    weighted_treatment = treatment * rank_weights
    
    if np.std(weighted_treatment) > 0 and np.std(weighted_outcome) > 0:
        rate = np.corrcoef(weighted_treatment, weighted_outcome)[0, 1]
    else:
        rate = np.nan
    
    return rate


# =============================================================================
# Binary Treatment Validation Functions (Uplift Curves)
# =============================================================================

def compute_uplift_curve(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100,
    normalize: bool = True
) -> UpliftCurveResult:
    """
    Compute uplift curve by ranking predictions and calculating cumulative uplift.
    
    The uplift curve shows the incremental effect of treatment on the outcome
    as we target increasingly larger portions of the population, ranked by
    predicted uplift score.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions (uplift scores). Higher values indicate higher
        predicted treatment effect.
    treatment : array-like
        Binary treatment indicator (1=treated, 0=control)
    outcome : array-like
        Observed outcome variable
    weights : array-like, optional
        Sample weights for weighted calculations
    n_bins : int, default=100
        Number of percentile bins for the curve
    normalize : bool, default=True
        If True, normalize uplift by population size
        
    Returns
    -------
    UpliftCurveResult
        Container with percentiles, uplift values, and group sizes
        
    Examples
    --------
    >>> predictions = model.predict(X)
    >>> result = compute_uplift_curve(predictions, treatment, outcome)
    >>> plt.plot(result.percentiles, result.cumulative_gain)
    """
    prediction, treatment, outcome, weights = _validate_inputs_binary(
        prediction, treatment, outcome, weights
    )
    
    # Sort by prediction (descending - highest uplift first)
    sort_idx = np.argsort(-prediction)
    prediction_sorted = prediction[sort_idx]
    treatment_sorted = treatment[sort_idx]
    outcome_sorted = outcome[sort_idx]
    weights_sorted = weights[sort_idx]
    
    n = len(prediction)
    percentiles = np.linspace(0, 1, n_bins + 1)[1:]  # Exclude 0
    
    uplift = np.zeros(n_bins)
    cumulative_gain = np.zeros(n_bins)
    n_treatment_arr = np.zeros(n_bins, dtype=int)
    n_control_arr = np.zeros(n_bins, dtype=int)
    
    for i, pct in enumerate(percentiles):
        cutoff_idx = int(np.ceil(pct * n))
        mask = np.zeros(n, dtype=bool)
        mask[:cutoff_idx] = True
        
        treat_rate, ctrl_rate, n_t, n_c = _compute_group_stats(
            outcome_sorted, treatment_sorted, weights_sorted, mask
        )
        
        n_treatment_arr[i] = n_t
        n_control_arr[i] = n_c
        
        if np.isnan(treat_rate) or np.isnan(ctrl_rate):
            uplift[i] = np.nan
            cumulative_gain[i] = np.nan
        else:
            uplift[i] = treat_rate - ctrl_rate
            if normalize:
                cumulative_gain[i] = uplift[i] * pct
            else:
                cumulative_gain[i] = uplift[i] * cutoff_idx
    
    # Random baseline (expected gain with random targeting)
    overall_treat_rate, overall_ctrl_rate, _, _ = _compute_group_stats(
        outcome_sorted, treatment_sorted, weights_sorted, np.ones(n, dtype=bool)
    )
    overall_uplift = overall_treat_rate - overall_ctrl_rate
    
    if normalize:
        random_baseline = percentiles * overall_uplift
    else:
        random_baseline = (percentiles * n) * overall_uplift
    
    return UpliftCurveResult(
        percentiles=percentiles,
        uplift=uplift,
        cumulative_gain=cumulative_gain,
        random_baseline=random_baseline,
        n_treatment=n_treatment_arr,
        n_control=n_control_arr
    )


def compute_qini_curve(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> UpliftCurveResult:
    """
    Compute Qini curve for uplift model evaluation.
    
    The Qini curve is similar to the uplift curve but measures the 
    incremental number of positive outcomes (not rate) as a function
    of the targeted population.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions (uplift scores)
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome (typically binary)
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of percentile bins
        
    Returns
    -------
    UpliftCurveResult
        Container with Qini curve values
    """
    prediction, treatment, outcome, weights = _validate_inputs_binary(
        prediction, treatment, outcome, weights
    )
    
    # Sort by prediction (descending)
    sort_idx = np.argsort(-prediction)
    treatment_sorted = treatment[sort_idx]
    outcome_sorted = outcome[sort_idx]
    weights_sorted = weights[sort_idx]
    
    n = len(prediction)
    percentiles = np.linspace(0, 1, n_bins + 1)[1:]
    
    qini_values = np.zeros(n_bins)
    n_treatment_arr = np.zeros(n_bins, dtype=int)
    n_control_arr = np.zeros(n_bins, dtype=int)
    
    # Overall treatment/control ratio for normalization
    n_treat_total = np.sum(treatment_sorted == 1)
    n_ctrl_total = np.sum(treatment_sorted == 0)
    
    for i, pct in enumerate(percentiles):
        cutoff_idx = int(np.ceil(pct * n))
        
        # Cumulative counts
        treat_mask = treatment_sorted[:cutoff_idx] == 1
        ctrl_mask = treatment_sorted[:cutoff_idx] == 0
        
        n_t = np.sum(treat_mask)
        n_c = np.sum(ctrl_mask)
        
        n_treatment_arr[i] = n_t
        n_control_arr[i] = n_c
        
        if n_t == 0 or n_c == 0:
            qini_values[i] = np.nan
        else:
            # Weighted sum of outcomes
            y_treat = np.sum(outcome_sorted[:cutoff_idx][treat_mask] * 
                           weights_sorted[:cutoff_idx][treat_mask])
            y_ctrl = np.sum(outcome_sorted[:cutoff_idx][ctrl_mask] * 
                          weights_sorted[:cutoff_idx][ctrl_mask])
            
            # Qini: Y_T - Y_C * (N_T / N_C) to account for imbalance
            qini_values[i] = y_treat - y_ctrl * (n_t / n_c)
    
    # Random baseline
    random_baseline = np.linspace(0, qini_values[-1] if not np.isnan(qini_values[-1]) else 0, n_bins)
    
    return UpliftCurveResult(
        percentiles=percentiles,
        uplift=qini_values,  # Store Qini values in uplift field
        cumulative_gain=qini_values,
        random_baseline=random_baseline,
        n_treatment=n_treatment_arr,
        n_control=n_control_arr
    )


def compute_cumulative_gain_curve(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> UpliftCurveResult:
    """
    Compute cumulative gain curve for uplift models.
    
    Shows the cumulative incremental gains from targeting the population
    in order of predicted uplift, compared to random targeting.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions (uplift scores)
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of percentile bins
        
    Returns
    -------
    UpliftCurveResult
        Container with cumulative gain values
    """
    return compute_uplift_curve(
        prediction, treatment, outcome, weights, n_bins, normalize=True
    )


# =============================================================================
# AUC Metric Functions
# =============================================================================

def compute_auuc(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> float:
    """
    Compute Area Under the Uplift Curve (AUUC).
    
    AUUC measures the overall quality of an uplift model by computing
    the area under the cumulative gain curve.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions (uplift scores)
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of bins for curve computation
        
    Returns
    -------
    float
        Area under the uplift curve
        
    Examples
    --------
    >>> auuc = compute_auuc(predictions, treatment, outcome)
    >>> print(f"AUUC: {auuc:.4f}")
    """
    curve = compute_uplift_curve(prediction, treatment, outcome, weights, n_bins)
    
    # Prepend (0, 0) point
    x = np.concatenate([[0], curve.percentiles])
    y = np.concatenate([[0], curve.cumulative_gain])
    
    # Remove NaN values
    valid_mask = ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]
    
    if len(x) < 2:
        return np.nan
    
    return integrate.trapezoid(y, x)


def compute_auuc_normalized(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> float:
    """
    Compute normalized AUUC (relative to random baseline).
    
    Normalized AUUC = (AUUC - Random_AUUC) / (Perfect_AUUC - Random_AUUC)
    
    Parameters
    ----------
    prediction : array-like
        Model predictions
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of bins
        
    Returns
    -------
    float
        Normalized AUUC between 0 and 1 (can be negative for bad models)
    """
    curve = compute_uplift_curve(prediction, treatment, outcome, weights, n_bins)
    
    # Model AUUC
    x = np.concatenate([[0], curve.percentiles])
    y_model = np.concatenate([[0], curve.cumulative_gain])
    y_random = np.concatenate([[0], curve.random_baseline])
    
    valid_mask = ~np.isnan(y_model) & ~np.isnan(y_random)
    x = x[valid_mask]
    y_model = y_model[valid_mask]
    y_random = y_random[valid_mask]
    
    if len(x) < 2:
        return np.nan
    
    auuc_model = integrate.trapezoid(y_model, x)
    auuc_random = integrate.trapezoid(y_random, x)
    
    # Perfect model would capture all uplift immediately
    overall_uplift = y_random[-1] if len(y_random) > 0 else 0
    auuc_perfect = overall_uplift  # Area of rectangle
    
    if auuc_perfect == auuc_random:
        return 0.0
    
    return (auuc_model - auuc_random) / (auuc_perfect - auuc_random)


def compute_qini_coefficient(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> float:
    """
    Compute the Qini coefficient.
    
    The Qini coefficient is the area between the Qini curve and the
    random baseline, normalized by the maximum possible area.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of bins
        
    Returns
    -------
    float
        Qini coefficient
    """
    curve = compute_qini_curve(prediction, treatment, outcome, weights, n_bins)
    
    x = np.concatenate([[0], curve.percentiles])
    y_model = np.concatenate([[0], curve.cumulative_gain])
    y_random = np.concatenate([[0], curve.random_baseline])
    
    valid_mask = ~np.isnan(y_model) & ~np.isnan(y_random)
    x = x[valid_mask]
    y_model = y_model[valid_mask]
    y_random = y_random[valid_mask]
    
    if len(x) < 2:
        return np.nan
    
    area_model = integrate.trapezoid(y_model, x)
    area_random = integrate.trapezoid(y_random, x)
    
    return area_model - area_random


def compute_all_auc_metrics(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> AUCResult:
    """
    Compute all AUC-based metrics for uplift model evaluation.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of bins
        
    Returns
    -------
    AUCResult
        Container with AUUC, normalized AUUC, Qini, and normalized Qini
    """
    auuc = compute_auuc(prediction, treatment, outcome, weights, n_bins)
    auuc_norm = compute_auuc_normalized(prediction, treatment, outcome, weights, n_bins)
    qini = compute_qini_coefficient(prediction, treatment, outcome, weights, n_bins)
    
    # Compute random AUUC for reference
    curve = compute_uplift_curve(prediction, treatment, outcome, weights, n_bins)
    x = np.concatenate([[0], curve.percentiles])
    y_random = np.concatenate([[0], curve.random_baseline])
    valid_mask = ~np.isnan(y_random)
    random_auuc = integrate.trapezoid(y_random[valid_mask], x[valid_mask])
    
    # Normalized Qini
    qini_curve = compute_qini_curve(prediction, treatment, outcome, weights, n_bins)
    max_qini = qini_curve.cumulative_gain[-1] if not np.isnan(qini_curve.cumulative_gain[-1]) else 1
    qini_norm = qini / max_qini if max_qini != 0 else 0
    
    return AUCResult(
        auuc=auuc,
        auuc_normalized=auuc_norm,
        qini=qini,
        qini_normalized=qini_norm,
        random_auuc=random_auuc
    )


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def compute_bootstrap_ci(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    metric_func: Callable,
    weights: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for an uplift metric.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    metric_func : callable
        Function that computes the metric (e.g., compute_auuc)
    weights : array-like, optional
        Sample weights
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for the interval
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, lower_bound, upper_bound)
    """
    # Don't enforce treatment type here - let the metric function handle it
    prediction = np.asarray(prediction).ravel()
    treatment = np.asarray(treatment).ravel()
    outcome = np.asarray(outcome).ravel()
    n = len(prediction)
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights).ravel()
    
    rng = np.random.RandomState(random_state)
    n = len(prediction)
    
    bootstrap_values = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        
        try:
            value = metric_func(
                prediction[idx],
                treatment[idx],
                outcome[idx],
                weights[idx]
            )
            if not np.isnan(value):
                bootstrap_values.append(value)
        except Exception:
            continue
    
    if len(bootstrap_values) < 10:
        return np.nan, np.nan, np.nan
    
    point_estimate = metric_func(prediction, treatment, outcome, weights)
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return point_estimate, lower, upper


# =============================================================================
# Complete Validation Function
# =============================================================================

def validate_causal_model(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    treatment_type: Optional[str] = None,
    n_bins: int = 100,
    compute_ci: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> ValidationResult:
    """
    Comprehensive validation of a causal/uplift model.
    
    This function computes all relevant curves and metrics for evaluating
    the quality of an uplift model, optionally with bootstrap confidence
    intervals. Supports BOTH binary and continuous treatments.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions (uplift scores for binary treatment, 
        CATE estimates for continuous treatment)
    treatment : array-like
        Treatment indicator (binary: 0/1) or treatment dose (continuous)
    outcome : array-like
        Observed outcome variable (typically binary)
    weights : array-like, optional
        Sample weights
    treatment_type : str, optional
        "binary" or "continuous". If None, auto-detected from data.
    n_bins : int, default=100
        Number of percentile bins for curves
    compute_ci : bool, default=False
        Whether to compute bootstrap confidence intervals
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level for intervals
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    ValidationResult
        Complete validation results including curves, metrics, and CIs
        
    Examples
    --------
    >>> # Binary treatment (uplift model)
    >>> result = validate_causal_model(
    ...     prediction=model.predict(X),
    ...     treatment=df['treatment'],  # 0 or 1
    ...     outcome=df['conversion'],
    ...     compute_ci=True
    ... )
    >>> print(f"AUUC: {result.auc_result.auuc:.4f}")
    
    >>> # Continuous treatment (dose-response model)
    >>> result = validate_causal_model(
    ...     prediction=model.predict(X),
    ...     treatment=df['dosage'],  # continuous values
    ...     outcome=df['response'],
    ...     treatment_type='continuous'
    ... )
    >>> print(f"Dose-response correlation: {result.continuous_metrics.dose_response_correlation:.4f}")
    """
    # Validate and detect treatment type
    prediction, treatment, outcome, weights, detected_type = _validate_inputs(
        prediction, treatment, outcome, weights, treatment_type
    )
    
    if detected_type == "binary":
        # Binary treatment: use uplift curves and AUUC metrics
        curve_result = compute_uplift_curve(
            prediction, treatment, outcome, weights, n_bins
        )
        
        auc_result = compute_all_auc_metrics(
            prediction, treatment, outcome, weights, n_bins
        )
        
        continuous_metrics = None
        
        # Optionally compute confidence intervals
        confidence_intervals = None
        if compute_ci:
            confidence_intervals = {}
            
            # AUUC CI
            _, auuc_lower, auuc_upper = compute_bootstrap_ci(
                prediction, treatment, outcome, compute_auuc,
                weights, n_bootstrap, confidence_level, random_state
            )
            confidence_intervals['auuc'] = (auuc_lower, auuc_upper)
            
            # Normalized AUUC CI
            _, auuc_norm_lower, auuc_norm_upper = compute_bootstrap_ci(
                prediction, treatment, outcome, compute_auuc_normalized,
                weights, n_bootstrap, confidence_level, random_state
            )
            confidence_intervals['auuc_normalized'] = (auuc_norm_lower, auuc_norm_upper)
            
            # Qini CI
            _, qini_lower, qini_upper = compute_bootstrap_ci(
                prediction, treatment, outcome, compute_qini_coefficient,
                weights, n_bootstrap, confidence_level, random_state
            )
            confidence_intervals['qini'] = (qini_lower, qini_upper)
    
    else:
        # Continuous treatment: use dose-response curves and correlation metrics
        curve_result = compute_dose_response_curve(
            prediction, treatment, outcome, weights, 
            n_bins=min(n_bins, 20)  # Fewer bins for continuous
        )
        
        auc_result = None
        
        continuous_metrics = compute_continuous_treatment_metrics(
            prediction, treatment, outcome, weights, n_bins=min(n_bins, 20)
        )
        
        # Confidence intervals for continuous metrics
        confidence_intervals = None
        if compute_ci:
            confidence_intervals = {}
            
            def _dose_response_corr(pred, treat, out, w):
                metrics = compute_continuous_treatment_metrics(pred, treat, out, w)
                return metrics.dose_response_correlation
            
            _, corr_lower, corr_upper = compute_bootstrap_ci(
                prediction, treatment, outcome, _dose_response_corr,
                weights, n_bootstrap, confidence_level, random_state
            )
            confidence_intervals['dose_response_correlation'] = (corr_lower, corr_upper)
            
            def _rank_corr(pred, treat, out, w):
                metrics = compute_continuous_treatment_metrics(pred, treat, out, w)
                return metrics.rank_correlation
            
            _, rank_lower, rank_upper = compute_bootstrap_ci(
                prediction, treatment, outcome, _rank_corr,
                weights, n_bootstrap, confidence_level, random_state
            )
            confidence_intervals['rank_correlation'] = (rank_lower, rank_upper)
    
    return ValidationResult(
        curve_result=curve_result,
        auc_result=auc_result,
        continuous_metrics=continuous_metrics,
        confidence_intervals=confidence_intervals,
        treatment_type=detected_type
    )


# =============================================================================
# Comparison and Visualization Utilities
# =============================================================================

def compare_models(
    predictions: Dict[str, np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 100
) -> pd.DataFrame:
    """
    Compare multiple uplift models using various metrics.
    
    Parameters
    ----------
    predictions : Dict[str, array-like]
        Dictionary mapping model names to their predictions
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
    n_bins : int, default=100
        Number of bins for curve computation
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each model
        
    Examples
    --------
    >>> predictions = {
    ...     'model_a': model_a.predict(X),
    ...     'model_b': model_b.predict(X),
    ...     'random': np.random.randn(len(X))
    ... }
    >>> comparison = compare_models(predictions, treatment, outcome)
    >>> print(comparison)
    """
    results = []
    
    for name, pred in predictions.items():
        auc_result = compute_all_auc_metrics(pred, treatment, outcome, weights, n_bins)
        results.append({
            'model': name,
            'auuc': auc_result.auuc,
            'auuc_normalized': auc_result.auuc_normalized,
            'qini': auc_result.qini,
            'qini_normalized': auc_result.qini_normalized,
            'lift_vs_random': auc_result.auuc - auc_result.random_auuc
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('auuc_normalized', ascending=False)
    
    return df


def get_uplift_by_decile(
    prediction: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute uplift statistics by prediction decile.
    
    Parameters
    ----------
    prediction : array-like
        Model predictions
    treatment : array-like
        Binary treatment indicator
    outcome : array-like
        Observed outcome
    weights : array-like, optional
        Sample weights
        
    Returns
    -------
    pd.DataFrame
        Uplift statistics for each decile
    """
    prediction, treatment, outcome, weights = _validate_inputs_binary(
        prediction, treatment, outcome, weights
    )
    
    # Create deciles
    deciles = pd.qcut(prediction, 10, labels=False, duplicates='drop')
    
    results = []
    for decile in sorted(np.unique(deciles), reverse=True):
        mask = deciles == decile
        
        treat_rate, ctrl_rate, n_t, n_c = _compute_group_stats(
            outcome, treatment, weights, mask
        )
        
        results.append({
            'decile': 10 - decile,  # 1 = highest predicted uplift
            'n_total': np.sum(mask),
            'n_treatment': n_t,
            'n_control': n_c,
            'treatment_rate': treat_rate,
            'control_rate': ctrl_rate,
            'uplift': treat_rate - ctrl_rate if not np.isnan(treat_rate) else np.nan,
            'avg_prediction': np.mean(prediction[mask])
        })
    
    return pd.DataFrame(results)


def plot_uplift_curves(
    result: Union[UpliftCurveResult, ValidationResult],
    title: str = "Uplift Curve",
    figsize: Tuple[int, int] = (10, 6),
    show_random: bool = True
) -> "matplotlib.figure.Figure":
    """
    Plot uplift curves with optional random baseline (for binary treatment).
    
    Parameters
    ----------
    result : UpliftCurveResult or ValidationResult
        Result from compute_uplift_curve or validate_causal_model
    title : str, default="Uplift Curve"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
    show_random : bool, default=True
        Whether to show random baseline
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    
    if isinstance(result, ValidationResult):
        if result.treatment_type != "binary":
            raise ValueError("Use plot_dose_response_curve for continuous treatment")
        curve = result.curve_result
    else:
        curve = result
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Cumulative gain curve
    ax1 = axes[0]
    ax1.plot(curve.percentiles, curve.cumulative_gain, 'b-', linewidth=2, label='Model')
    if show_random:
        ax1.plot(curve.percentiles, curve.random_baseline, 'r--', linewidth=1.5, label='Random')
    ax1.fill_between(curve.percentiles, curve.random_baseline, curve.cumulative_gain, 
                     alpha=0.3, where=curve.cumulative_gain >= curve.random_baseline)
    ax1.set_xlabel('Population Fraction')
    ax1.set_ylabel('Cumulative Gain')
    ax1.set_title('Cumulative Gain Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Uplift by percentile
    ax2 = axes[1]
    ax2.plot(curve.percentiles, curve.uplift, 'g-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Population Fraction')
    ax2.set_ylabel('Uplift')
    ax2.set_title('Uplift by Population Percentile')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_dose_response_curve(
    result: Union[DoseResponseResult, ValidationResult],
    title: str = "Dose-Response Curve",
    figsize: Tuple[int, int] = (10, 6),
    show_error_bars: bool = True
) -> "matplotlib.figure.Figure":
    """
    Plot dose-response curve for continuous treatment validation.
    
    Parameters
    ----------
    result : DoseResponseResult or ValidationResult
        Result from compute_dose_response_curve or validate_causal_model
    title : str, default="Dose-Response Curve"
        Plot title
    figsize : tuple, default=(10, 6)
        Figure size
    show_error_bars : bool, default=True
        Whether to show standard error bars
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    
    if isinstance(result, ValidationResult):
        if result.treatment_type != "continuous":
            raise ValueError("Use plot_uplift_curves for binary treatment")
        curve = result.curve_result
    else:
        curve = result
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Dose-response curve
    ax1 = axes[0]
    valid_mask = ~np.isnan(curve.observed_response)
    
    if show_error_bars:
        ax1.errorbar(
            curve.bin_centers[valid_mask], 
            curve.observed_response[valid_mask],
            yerr=curve.std_error[valid_mask],
            fmt='o-', capsize=3, label='Observed', color='blue'
        )
    else:
        ax1.plot(curve.bin_centers[valid_mask], curve.observed_response[valid_mask], 
                'o-', label='Observed', color='blue')
    
    ax1.plot(curve.bin_centers[valid_mask], curve.predicted_effect[valid_mask], 
            's--', label='Predicted', color='orange')
    ax1.set_xlabel('Treatment Dose')
    ax1.set_ylabel('Response Rate')
    ax1.set_title('Dose-Response Relationship')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calibration plot (predicted vs observed)
    ax2 = axes[1]
    ax2.scatter(curve.predicted_effect[valid_mask], curve.observed_response[valid_mask], 
               s=curve.n_samples[valid_mask] / 10, alpha=0.7)
    
    # Perfect calibration line
    lims = [
        min(ax2.get_xlim()[0], ax2.get_ylim()[0]),
        max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel('Predicted Effect')
    ax2.set_ylabel('Observed Response')
    ax2.set_title('Calibration Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =============================================================================
# Pandas DataFrame Integration
# =============================================================================

def validate_from_dataframe(
    df: pd.DataFrame,
    prediction_col: str,
    treatment_col: str,
    outcome_col: str,
    weight_col: Optional[str] = None,
    **kwargs
) -> ValidationResult:
    """
    Validate a causal model directly from a pandas DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    prediction_col : str
        Name of column with model predictions
    treatment_col : str
        Name of column with treatment indicator
    outcome_col : str
        Name of column with outcome variable
    weight_col : str, optional
        Name of column with sample weights
    **kwargs
        Additional arguments passed to validate_causal_model
        
    Returns
    -------
    ValidationResult
        Complete validation results
        
    Examples
    --------
    >>> result = validate_from_dataframe(
    ...     df,
    ...     prediction_col='uplift_score',
    ...     treatment_col='treated',
    ...     outcome_col='converted'
    ... )
    """
    weights = df[weight_col].values if weight_col else None
    
    return validate_causal_model(
        prediction=df[prediction_col].values,
        treatment=df[treatment_col].values,
        outcome=df[outcome_col].values,
        weights=weights,
        **kwargs
    )


# =============================================================================
# Main entry point for CLI usage
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # =========================================================================
    # DEMO 1: Binary Treatment Validation
    # =========================================================================
    print("=" * 70)
    print("DEMO 1: Binary Treatment (Uplift Model) Validation")
    print("=" * 70)
    
    n = 5000
    
    # Simulate data with binary treatment
    X = np.random.randn(n, 5)
    treatment_binary = np.random.binomial(1, 0.5, n)
    true_uplift = 0.1 + 0.2 * X[:, 0] + 0.1 * X[:, 1]
    base_prob = 0.3 + 0.1 * X[:, 2]
    outcome_binary = np.random.binomial(
        1, np.clip(base_prob + treatment_binary * true_uplift, 0, 1)
    )
    
    # Simulate model predictions (correlated with true uplift)
    prediction_binary = true_uplift + np.random.randn(n) * 0.1
    
    # Run validation
    result_binary = validate_causal_model(
        prediction=prediction_binary,
        treatment=treatment_binary,
        outcome=outcome_binary,
        compute_ci=True,
        n_bootstrap=500,
        random_state=42
    )
    
    print(f"\nTreatment Type Detected: {result_binary.treatment_type}")
    print("\nAUC Metrics:")
    print(f"  AUUC:            {result_binary.auc_result.auuc:.4f}")
    print(f"  AUUC Normalized: {result_binary.auc_result.auuc_normalized:.4f}")
    print(f"  Qini:            {result_binary.auc_result.qini:.4f}")
    print(f"  Qini Normalized: {result_binary.auc_result.qini_normalized:.4f}")
    print(f"  Random AUUC:     {result_binary.auc_result.random_auuc:.4f}")
    
    if result_binary.confidence_intervals:
        print("\n95% Confidence Intervals:")
        for metric, (lower, upper) in result_binary.confidence_intervals.items():
            print(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
    
    # Decile analysis
    print("\nUplift by Decile:")
    decile_df = get_uplift_by_decile(prediction_binary, treatment_binary, outcome_binary)
    print(decile_df.to_string(index=False))
    
    # =========================================================================
    # DEMO 2: Continuous Treatment Validation
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO 2: Continuous Treatment (Dose-Response Model) Validation")
    print("=" * 70)
    
    # Simulate data with continuous treatment (dosage)
    treatment_continuous = np.random.uniform(0, 10, n)  # Dosage from 0 to 10
    
    # True dose-response: diminishing returns
    true_effect = 0.1 + 0.05 * treatment_continuous - 0.002 * treatment_continuous ** 2
    true_effect += 0.02 * X[:, 0] * treatment_continuous  # Heterogeneous effect
    
    outcome_continuous = np.random.binomial(
        1, np.clip(0.2 + true_effect + 0.05 * X[:, 2], 0, 1)
    )
    
    # Model predictions (CATE estimates)
    prediction_continuous = true_effect + np.random.randn(n) * 0.05
    
    # Run validation (auto-detects continuous treatment)
    result_continuous = validate_causal_model(
        prediction=prediction_continuous,
        treatment=treatment_continuous,
        outcome=outcome_continuous,
        compute_ci=True,
        n_bootstrap=500,
        random_state=42
    )
    
    print(f"\nTreatment Type Detected: {result_continuous.treatment_type}")
    print("\nContinuous Treatment Metrics:")
    cm = result_continuous.continuous_metrics
    print(f"  Dose-Response Correlation: {cm.dose_response_correlation:.4f}")
    print(f"  Rank Correlation (Spearman): {cm.rank_correlation:.4f}")
    print(f"  Kendall's Tau: {cm.kendall_tau:.4f}")
    print(f"  MSE: {cm.mse:.4f}")
    print(f"  MAE: {cm.mae:.4f}")
    print(f"  Calibration Slope: {cm.calibration_slope:.4f}")
    print(f"  Calibration Intercept: {cm.calibration_intercept:.4f}")
    
    if result_continuous.confidence_intervals:
        print("\n95% Confidence Intervals:")
        for metric, (lower, upper) in result_continuous.confidence_intervals.items():
            print(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
    
    # Treatment effect by quantile
    print("\nTreatment Effect by Prediction Quantile:")
    quantile_df = compute_treatment_effect_by_quantile(
        prediction_continuous, treatment_continuous, outcome_continuous
    )
    print(quantile_df.to_string(index=False))
    
    # RATE metric
    rate = compute_grf_rank_weighted_average(
        prediction_continuous, treatment_continuous, outcome_continuous
    )
    print(f"\nRank-Weighted Average Treatment Effect (RATE): {rate:.4f}")
    
    # =========================================================================
    # DEMO 3: Model Comparison (Binary Treatment)
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMO 3: Model Comparison")
    print("=" * 70)
    
    predictions_dict = {
        'trained_model': prediction_binary,
        'random_baseline': np.random.randn(n),
        'perfect_model': true_uplift
    }
    comparison = compare_models(predictions_dict, treatment_binary, outcome_binary)
    print("\nBinary Treatment Model Comparison:")
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Validation complete!")
    print("=" * 70)
