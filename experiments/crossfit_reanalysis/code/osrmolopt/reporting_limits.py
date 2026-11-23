"""Numerical implementation extracted from the frozen experiment source."""
from __future__ import annotations
import copy, json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import binomtest, spearmanr, beta
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import Ridge, TheilSenRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
SEED = 20261117



def potency_interval(kd: float) -> tuple[float, float]:
    if kd == 10000.0:
        return (-math.inf, 5.0)
    value = 9.0 - math.log10(kd)
    return (value, value)

def linear_interval(terms: list[tuple[float, tuple[float, float]]]) -> tuple[float, float]:
    lower = upper = 0.0
    for coefficient, (lo, hi) in terms:
        if coefficient > 0:
            lower += coefficient * lo
            upper += coefficient * hi
        else:
            lower += coefficient * hi
            upper += coefficient * lo
    return (lower, upper)

def orient(interval: tuple[float, float], sign: float) -> tuple[float, float]:
    return interval if sign > 0 else (-interval[1], -interval[0])

def classify(interval: tuple[float, float]) -> str:
    if interval[0] > 0:
        return 'positive'
    if interval[1] < 0:
        return 'negative'
    return 'indeterminate'

def json_float(value: float):
    if math.isinf(value):
        return '-inf' if value < 0 else '+inf'
    return float(value)
