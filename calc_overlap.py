import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

def compute_gps_overlap(df, treatment_col, covariate_cols):
    """
    Calcula métricas de overlap/positivity para tratamento contínuo
    usando o Generalized Propensity Score (GPS).
    
    Retorna dict com métricas e o array de GPS por observação.
    """
    X = df[covariate_cols].values
    T = df[treatment_col].values

    # 1) Ajustar modelo T | X (pode trocar por algo mais flexível)
    model = LinearRegression().fit(X, T)
    T_pred = model.predict(X)
    residuals = T - T_pred
    sigma = residuals.std()

    # 2) GPS = f(T_i | X_i) sob normalidade
    gps = norm.pdf(T, loc=T_pred, scale=sigma)

    # 3) Pesos IPW (inverse of GPS) normalizados
    weights = 1.0 / np.clip(gps, 1e-10, None)
    weights_norm = weights / weights.sum()

    # 4) Métricas de overlap
    # ESS: quanto menor relativo a N, pior o overlap
    ess = 1.0 / np.sum(weights_norm ** 2)
    ess_ratio = ess / len(T)  # idealmente próximo de 1

    # Coeficiente de variação dos pesos (menor = melhor)
    cv_weights = weights.std() / weights.mean()

    # Percentis do GPS (valores muito baixos indicam violação de positivity)
    gps_percentiles = np.percentile(gps, [1, 5, 10, 25, 50])

    # Fração de unidades com GPS < threshold (potenciais violações)
    threshold = np.percentile(gps, 5)  # ou um valor absoluto
    frac_low_gps = np.mean(gps < threshold)

    metrics = {
        "n": len(T),
        "ess": round(ess, 1),
        "ess_ratio": round(ess_ratio, 4),       # → 1 = bom
        "cv_weights": round(cv_weights, 4),      # → 0 = bom
        "gps_p1": round(gps_percentiles[0], 6),
        "gps_p5": round(gps_percentiles[1], 6),
        "gps_p10": round(gps_percentiles[2], 6),
        "gps_p25": round(gps_percentiles[3], 6),
        "gps_median": round(gps_percentiles[4], 6),
        "sigma_T_given_X": round(sigma, 4),
    }

    return metrics, gps

covariates = ["faixa_idade", "prazo", "score_credito", "regiao"]
metrics, gps = compute_gps_overlap(df, treatment_col="preco", covariate_cols=covariates)

print(metrics)
# {'n': 50000, 'ess': 42150.3, 'ess_ratio': 0.843, 'cv_weights': 0.412, ...}